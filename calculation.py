import numpy as np
from numba import jit

@jit
def putinto3by3(a,b,c,d,e,f,g,h,i): # populate a 3*3 matrix
    arr = np.zeros((3,3))
    arr[0,0] = a
    arr[0,1] = b
    arr[0,2] = c
    arr[1,0] = d
    arr[1,1] = e
    arr[1,2] = f
    arr[2,0] = g
    arr[2,1] = h
    arr[2,2] = i
    return arr

@jit
def angular(r: np.float64, rsep: np.float64, rp: np.float64, b: np.float64, phase: np.float64, sample: np.int64):
    """
    Describes the angular shape of the probed region for a particular phase

    Args:
        r: atmosphere radius
        rsep: orbital radius
        rp: planet's radius
        b: impact parameter
        phase: phase
        sample: controls the number of points sampled
    
    Returns:
        polar and azimuthal angles arrays
    """

    # first start in 2D cartesian coordinates, the projection of the star is a unit circle and the transit is horizonal left-to-right

    incline = np.arcsin(b/rsep)
    x0 = rsep*np.sin(2*np.pi*phase)
    y0 = rsep*np.sin(incline)*np.cos(2*np.pi*phase) # the planet's position
    intersect = True # check intersections of the projections of the star and the atmosphere on this 2D plane
    if x0 == 0 and y0 == 0:
        intersect = False
    elif x0 == 0 and y0 != 0:
        y = (1 - r**2 + y0**2) / (2*y0)
        if abs(y) >= 1:
            intersect = False
        else:
            x1 = np.sqrt(1 - y**2)
            x2 = -np.sqrt(1 - y**2)
            y1 = y
            y2 = y
    elif y0 == 0 and x0 != 0:
        x = (1 - r**2 + x0**2)/ (2*x0)
        if abs(x) >= 1:
            intersect = False
        else:
            y1 = np.sqrt(1-x**2)
            y2 = -np.sqrt(1-x**2)
            x1 = x
            x2 = x
    else:
        smile = (1 + x0**2 + y0**2 - r**2)/(2*y0)
        delta = 4*(x0*smile/y0)**2 - 4*(1+(x0/y0)**2)*(smile**2-1)
        if delta <= 0:
            intersect = False
        else:
            x1 = (2*(x0*smile/y0) + np.sqrt(delta)) / (2*(1 + (x0/y0)**2))
            x2 = (2*(x0*smile/y0) - np.sqrt(delta)) / (2*(1 + (x0/y0)**2))
            y1 = smile - x1*x0/y0
            y2 = smile - x2*x0/y0

    # still in the flat 2D picture, these intersection points (or the lack of) determine the segment probed in the atmosphere
    # now consider a 3D cartesian coordinate system centred at the planet, (also induce spherical polars)
    # with the planet travelling in the xz-plane (increasing x) and the z-axis pointing to the observer
    # the "shadow cast by planet" on the atmosphere will be two disks, both parallel to the xy-plane
    # restricting the longitude (azimuthal angle) to describe the very narrow terminator
    # in this frame, the region is approximately covered angularly by the circular strip formed by 
    # the cartesian product of the azimuth interval and the polar extent determined by the segment.

    halfwidth = np.sqrt(r**2 - rp**2)
    if not intersect:
        if x0**2 + y0**2 > 1:
            return np.array([10.0]), np.array([10.0])
        else:
            tmin = 0
            tmax = 2*np.pi # either nothing or full θ range if no intersection
    else:
        x1 = x1-x0
        y1 = y1-y0
        x2 = x2-x0
        y2 = y2-y0
        angle1 = np.atan2(y1,x1)
        angle2 = np.atan2(y2,x2)
        if angle1 < 0:
            angle1 += 2*np.pi
        if angle2 < 0:
            angle2 += 2*np.pi
        big = max(angle1, angle2)
        small = min(angle1, angle2)
        if y1 > 0 and y2 > 0: # carefully define the polar extent
            if y0 < 0:
                tmin = small
                tmax = big
            else:
                tmin = big
                tmax = small
        elif y1 < 0 and y2 < 0:
            if y0 < 0:
                tmin = big
                tmax = small
            else:
                tmin = small
                tmax = big
        elif (x1 < 0 and x2 < 0) or (x1 > 0 and x2 > 0):
            if x0 < 0:
                tmin = big
                tmax = small
            else:
                tmin = small
                tmax = big
        elif small >= np.pi/2 or big <= 3*np.pi/2:
            if x0 < 0:
                tmin = big
                tmax = small
            else:
                tmin = small
                tmax = big
        elif big <= 3*np.pi/2:
            if x0 < 0:
                tmin = small
                tmax = big
            else:
                tmin = big
                tmax = small
        else:
            raise Exception("bug encountered")

    # construct the circular band in the same 3D cartesian system (note planet at origin)
    if tmax > tmin:
        sample_along = int(sample*(tmax - tmin)*rp/(2*halfwidth))
    else:
        sample_along = int(sample*(tmax + 2*np.pi - tmin)*rp/(2*halfwidth))
    x_coordinates = np.zeros((sample, sample_along))
    y_coordinates = np.zeros((sample, sample_along))
    z_coordinates = np.zeros((sample, sample_along))
    for n, i in enumerate(np.linspace(-halfwidth, halfwidth, sample)):
        z_coordinates[n,:] = np.ones(sample_along)*i
        if tmax > tmin:
            x_coordinates[n,:] = rp*np.cos(np.linspace(tmin, tmax, sample_along))
            y_coordinates[n,:] = rp*np.sin(np.linspace(tmin, tmax, sample_along))
        else:
            division = int(sample_along*tmax/(tmax+2*np.pi - tmin))
            x_coordinates[n,:division] = rp*np.cos(np.linspace(0, tmax, division))
            x_coordinates[n, division:sample_along] = rp*np.cos(np.linspace(tmin, 2*np.pi, sample_along-division))
            y_coordinates[n,:division] = rp*np.sin(np.linspace(0, tmax, division))
            y_coordinates[n, division:sample_along] = rp*np.sin(np.linspace(tmin, 2*np.pi, sample_along-division))

    # the 3D cartesian system has the z-axis pointing to the observer and x-axis to the right hand side of the observer, and planet at origin
    # attach new x and z axes to the planet, we perform rotations so to make them align with the observer's axes
    # first consider de-inclination

    de_incline = putinto3by3(1.0, 0, 0,0, np.cos(incline), -np.sin(incline), 0, np.sin(incline), np.cos(incline))
    x_axis = np.array([-np.sin(2*np.pi*phase), 0.0, -np.cos(2*np.pi*phase)]) # substellar x-axis after de-inclination 
    y_angle = np.arctan2(x_axis[2], x_axis[0])  # rotation about true y-axis to make planet x-axis align
    y_rotate = putinto3by3(np.cos(y_angle),0.0 , np.sin(y_angle),0, 1, 0,-np.sin(y_angle), 0, np.cos(y_angle))
    x_rotate = putinto3by3(1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0) # finally rotate about x-axis by π/2
    matrix =  np.dot(x_rotate, np.dot(y_rotate, de_incline))
    # then just apply the rotations and extract the angular coordinates
    theta = np.zeros((sample, sample_along))
    phi = np.zeros((sample, sample_along))
    for i in range(sample):
        for j in range(sample_along):
            x, y, z = np.dot(matrix,np.array([x_coordinates[i,j], y_coordinates[i,j], z_coordinates[i,j]]))
            theta[i,j] = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))
            if y >= 0:
                phi[i,j] = np.arccos(x/np.sqrt(x**2 + y**2))
            else:
                phi[i,j] = -np.arccos(x/np.sqrt(x**2 + y**2))
    return theta.flatten(), phi.flatten()

@jit
def morning_evening(max_phase, total_points, H, rp, rsep, b, sample_points = 40):
    """
    Calculate the morning and evening contributions.
    Args:
        max_phase: the maximum phase
        total_points: total number of samples to take within +/- max_phase
        H: atmospheric scale height
        rp: planet radius
        rsep: orbital radius or semi-major axis
        b: impact parameter
        sample_points: number of sample points along the width of the strip 

    Returns:
        array containing phase values

        array of morning contribution

        array of evening contribution

        phase when the evening contribution first emerge

        phase when the morning contribution disappear at last
    """
    p = np.linspace(-max_phase, max_phase, total_points)
    signal_m = np.zeros(total_points)
    signal_e = np.zeros(total_points)
    for n, x in enumerate(p):
        theta, phi = angular(H+rp, rsep, rp, b, x, sample_points)
        phi = phi[phi!=10.0]
        l = len(phi)
        if l != 0:
            big = np.sum(phi>=0)
            signal_e[n] = big
            signal_m[n] = l - big
    return p, signal_m, signal_e, [signal_m!=0][0], [signal_e!=0][-1]

@jit
def long_distribution(phase, n_bins, H, rp, rsep, b, sample_points = 40):
    """
    Longitudinal distribution, gives a histogram of frequencies of points falling within longiude boundaries

    Args:
        phase: phase
        n_bins: number of bins
        H: atmospheric scale height
        rp: planet radius
        rsep: orbital radius
        b: impact parameter
    """

    hist  = np.zeros(n_bins)
    bounds = np.linspace(-np.pi, np.pi, n_bins+1)
    theta, phi = angular(H+rp, rsep, rp, b, phase, sample_points)
    for i in n_bins:
        l = phi[phi>bounds[i]]
        hist[i] = np.sum(l < bounds[i+1])
    return hist