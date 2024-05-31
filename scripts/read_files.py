import numpy as np
import h5py

def read_file(path, isq3d=False, mirror=False):

    """
    Reads an osiris file, whatever its dimension
    """

    # Open the file
    with h5py.File(path,'r') as f:

        f_keys = list( f.keys() )
        data = f.__getitem__( f_keys[-1] )
        Ndims = data.ndim

    # Call the adequate routine
    if Ndims == 1:
        output = read_file_1d(path)

    elif Ndims == 2:
        if isq3d :
            output = read_file_q3d(path, mirror)
        else:
            output = read_file_2d(path)

    elif Ndims == 3:
        output = read_file_3d(path)

    else:
        output = 0.0
        raise("Error Number of dimemsions must be 1,2 or 3")

    return output

def read_file_1d(path):

    """
    Reads a 1d osiris file
    Returns min_x, max_x, N_x, fdata
    """    

    # Open the file
    with h5py.File(path,'r') as f:

        # Get keys of the file
        f_keys = list( f.keys() )
        data = f.__getitem__( f_keys[-1] )
        N_x = data.shape[0]

        # Read the array and store it
        fdata = np.zeros([N_x])
        data.read_direct(fdata, np.s_[0:N_x], np.s_[0:N_x])

        # Get the mins and maxs bounds along each direction
        min_x, max_x = f['AXIS/AXIS1'][:]

    return min_x, max_x, N_x, fdata

def read_file_2d(path):

    """
    Reads a 2d osiris file
    Returns min_x, max_x, N_x, min_y, max_y, N_y, fdata
    """    

    # Open the file
    with h5py.File(path,'r') as f:

        # Get keys of the file
        f_keys = list( f.keys() )
        data = f.__getitem__( f_keys[-1] )
        N_y, N_x = data.shape[0], data.shape[1]

        # Read the array and store it
        fdata = np.zeros([N_y, N_x])
        data.read_direct(fdata, np.s_[0:N_y,0:N_x], np.s_[0:N_y,0:N_x])

        # Get the mins and maxs bounds along each direction
        min_x, max_x = f['AXIS/AXIS1'][:]
        min_y, max_y = f['AXIS/AXIS2'][:]

    return min_x, max_x, N_x, min_y, max_y, N_y, fdata

def read_file_3d(path):

    """
    Reads a 3d osiris file
    Returns min_x, max_x, N_x, min_y, max_y, N_y, min_z, max_z, N_z, fdata
    """    

    # Open the file
    with h5py.File(path,'r') as f:

        # Get keys of the file
        f_keys = list( f.keys() )
        data = f.__getitem__( f_keys[-1] )
        N_z, N_y, N_x = data.shape[0], data.shape[1], data.shape[2]

        # Read the array and store it
        fdata = np.zeros([N_z, N_y, N_x])
        data.read_direct(fdata, np.s_[0:N_z,0:N_y,0:N_x], np.s_[0:N_z,0:N_y,0:N_x])

        # Get the mins and maxs bounds along each direction
        min_x, max_x = f['AXIS/AXIS1'][:]
        min_y, max_y = f['AXIS/AXIS2'][:]
        min_z, max_z = f['AXIS/AXIS3'][:]

    return min_x, max_x, N_x, min_y, max_y, N_y, min_z, max_z, N_z, fdata

def read_file_q3d(path, mirror):

    """
    Reads a quasi-3D osiris file
    And mirors the data with respect to the radial axis
    Returns min_x, max_x, N_x, min_y, max_y, N_y, fdata
    """    

    # We read the 2d file
    min_x, max_x, N_x, min_y, max_y, N_y, data_tmp = read_file_2d(path)

    # We mirror the data along the axis r (0 for python)
    # We don't care about the 0th cell because it is at -dr/2.  We'll get that in the theta + pi part.
    if mirror :
        fdata = np.concatenate( (np.flip(data_tmp[1:,:],axis=0), data_tmp[1:,:] ), axis=0 )
        min_y = - max_y
        N_y = 2 * ( N_y - 1)
    else:
        fdata = data_tmp.copy()

    return min_x, max_x, N_x, min_y, max_y, N_y, fdata

def sum_flds_over_modes(path, key, n_modes, n, theta, mirror):

    """
    Reads the specified field and sum it over the modes
    path (string) path to simulation folder
    key (string) fields we want to ploit (e2_cyl_m for example)
    n_modes (int) number of modes in the simulation
    n (int) iteration we want to plot
    theta (double) azimuthal slice we want to plot (only if 1 in m)
    mirror (boolean) mirros the data with respect to r axis
    """    

    # first entry will always stay None
    f_real = [None]*(n_modes+1)
    f_imag = [None]*(n_modes+1)

    # Reads for mode 0
    file_path = path + "MS/FLD/MODE-0-RE/" + key + "/" + key + '-{:d}-re-{:06d}.h5'.format(0,n)
    min_x, max_x, N_x, min_y, max_y, N_y, f_real[0] = read_file(file_path, True, mirror)

    # data summed over all modes
    fdata = f_real[0]

    # Read and add other modes
    for m in np.arange(1,n_modes+1):

        # Real part
        file_path = path + "MS/FLD/MODE-1-RE/" + key + "/" + key + '-{:d}-re-{:06d}.h5'.format(m,n)
        min_x, max_x, N_x, min_y, max_y, N_y, f_real[m] = read_file(file_path, True, mirror)
        fdata += np.cos( m * ( theta + np.pi ) ) * f_real[m]

        # Imaginary part
        file_path = path + "MS/FLD/MODE-1-IM/" + key + "/" + key + '-{:d}-im-{:06d}.h5'.format(m,n)
        min_x, max_x, N_x, min_y, max_y, N_y, f_imag[m] = read_file(file_path, True, mirror)
        fdata += np.sin( m * ( theta + np.pi ) ) * f_imag[m]

    return min_x, max_x, N_x, min_y, max_y, N_y, fdata

def sum_dens_over_modes(path, spc, key, n_modes, n, theta, mirror):

    """
    Reads the specified density and sum it over the modes
    path (string) path to simulation folder
    spc (string) specie name
    key (string) fields we want to ploit (charge_cyl_m-savg for example)
    n_modes (int) number of modes in the simulation
    n (int) iteration we want to plot
    theta (double) azimuthal slice we want to plot (only if 1 in m)
    mirror (boolean) mirros the data with respect to r axis
    """    

    # first entry will always stay None
    f_real = [None]*(n_modes+1)
    f_imag = [None]*(n_modes+1)

    # Reads for mode 0
    file_path = path + "MS/DENSITY/" + spc +  "/MODE-0-RE/" + key + "/" + key + "-" + spc + '-{:d}-re-{:06d}.h5'.format(0,n)
    min_x, max_x, N_x, min_y, max_y, N_y, f_real[0] = read_file(file_path, True, mirror)

    # data summed over all modes
    fdata = f_real[0]

    # Read and add other modes
    for m in np.arange(1,n_modes+1):

        # Real part
        file_path = path + "MS/DENSITY/" + spc +  "/MODE-1-RE/" + key + "/" + key + "-" + spc + '-{:d}-re-{:06d}.h5'.format(m,n)
        min_x, max_x, N_x, min_y, max_y, N_y, f_real[m] = read_file(file_path, True, mirror)
        fdata += np.cos( m * ( theta + np.pi ) ) * f_real[m]

        # Imaginary part
        file_path = path + "MS/DENSITY/" + spc +  "/MODE-1-IM/" + key + "/" + key + "-" + spc + '-{:d}-im-{:06d}.h5'.format(m,n)
        min_x, max_x, N_x, min_y, max_y, N_y, f_imag[m] = read_file(file_path, True, mirror)
        fdata += np.sin( m * ( theta + np.pi ) ) * f_imag[m]

    return min_x, max_x, N_x, min_y, max_y, N_y, fdata

def read_file_q3d_concatenate(path_01, path_02):

    """
    Reads two quasi-3D osiris file
    And mirors their data with respect to the radial axis
    The first file is on the upper half
    The second file is on the lower half
    Returns min_x, max_x, N_x, min_y, max_y, N_y, fdata
    """    

    # We read the 2 files, they can have different grids
    min_x, max_x, N_x, min_y, max_y, N_y, data_tmp_01 = read_file_2d(path_01)
    min_x, max_x, N_x, min_y, max_y, N_y, data_tmp_02 = read_file_2d(path_02)

    # We concatenate
    fdata = np.concatenate( (np.flip(data_tmp_02[1:,:],axis=0), data_tmp_01[1:,:] ), axis=0 )
    
    # We adapt the bounds
    min_y = - max_y
    N_y = 2 * ( N_y - 1)

    return min_x, max_x, N_x, min_y, max_y, N_y, fdata

def read_file_q3d_flip(path):

    """
    Reads a quasi-3D file and flips it
    """    

    # We read the 2 files, they can have different grids
    min_x, max_x, N_x, min_y, max_y, N_y, data_tmp = read_file_2d(path)

    fdata = np.flip(data_tmp[:,:],axis=0)
    var_tmp = max_y
    min_y_new = - max_y
    max_y_new = - min_y
    
    return min_x, max_x, N_x, min_y_new, max_y_new, N_y, fdata

def read_raw(path, quants):

    data_raw = dict()

    # Loading the data and closing the file
    with h5py.File( path, 'r' ) as f:

        # We loop on the quantities in quants
        for key in quants:
            is_empty = ( f.__getitem__(key).shape==() )
            if not is_empty:
                data_raw[key] = f.__getitem__(key)[:]
                N_raw = len(data_raw[key])
            else:
                data_raw[key] = list()
                N_raw = 0

        # Convert list to array
        data_raw[key] = np.array(data_raw[key])

    return data_raw

def read_field_envelope(path):

    # 2D field
    min_x, max_x, N_x, min_y, max_y, N_y, signal = read_file(path)
    dx = (max_x - min_x) / N_x
    dy = (max_y - min_y) / N_y

    # Fourrier transform
    signal_fft = np.fft.fftshift( np.fft.fft2(signal) )

    # Frequencies
    kx = 2. * np.pi * np.fft.fftshift( np.fft.fftfreq(N_x, d=dx) )

    # Filter, empirical limit that works in 2D and 3D
    klim = 0.5**2
    high_freq_signal = signal_fft.copy()
    mask = (kx>klim)
    high_freq_signal[:,mask] = 0.0

    # IFFT of the laser envelope
    dataset_ifft = np.fft.ifft2(np.fft.ifftshift(high_freq_signal))
    high_freq_signal_pow = np.abs(dataset_ifft)**2
    high_freq_signal_pow *= np.amax(signal) / np.amax(high_freq_signal_pow)

    return min_x, max_x, N_x, min_y, max_y, N_y, high_freq_signal_pow

