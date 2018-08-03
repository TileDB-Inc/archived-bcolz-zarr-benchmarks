import bcolz
# import matplotlib.pyplot as plt
import numpy as np
import os
import s3fs
import shutil
import subprocess
import sys
import tiledb
import time
import zarr

# Set up global variables/configs
enable_s3 = False
tiledb_array_name = 'tiledb_array'
bcolz_array_name = 'bcolz_array'
zarr_array_name = 'zarr_array'
s3_bucket_name = 'tiledb-bench'
bcolz.defaults.cparams['cname'] = 'blosclz'
bcolz.defaults.cparams['clevel'] = 4
bcolz.defaults.eval_vm = 'numexpr'
config = tiledb.Config()
config['sm.tile_cache_size'] = 100 * 1024 * 1024  # 100MB

if enable_s3:
    zarr_s3 = s3fs.S3FileSystem(key=os.environ['AWS_ACCESS_KEY_ID'],
                                secret=os.environ['AWS_SECRET_ACCESS_KEY'])
    zarr_s3_store = s3fs.S3Map(root=s3_bucket_name + '/' + zarr_array_name,
                               s3=zarr_s3, check=False)

def sync_fs():
    subprocess.call(['sudo', 'sync'])


def drop_fs_cache():
    # subprocess.call(['sudo', 'purge'])
    subprocess.call(['sudo', 'sh', '-c', 'echo 3 >/proc/sys/vm/drop_caches'])


def remove_arrays():
    if enable_s3:
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx)
        tiledb_path = 's3://{}/{}'.format(s3_bucket_name, tiledb_array_name)
        zarr_path = 's3://{}/{}'.format(s3_bucket_name, zarr_array_name)
        if vfs.is_dir(tiledb_path):
            tiledb.remove(tiledb.Ctx(), tiledb_path)
        if vfs.is_dir(zarr_path):
            vfs.remove_dir(zarr_path)
    else:
        if os.path.exists(tiledb_array_name):
            tiledb.remove(tiledb.Ctx(), tiledb_array_name)
        if os.path.exists(bcolz_array_name):
            shutil.rmtree(bcolz_array_name)
        if os.path.exists(zarr_array_name):
            shutil.rmtree(zarr_array_name)

def create_tiledb_array(ctx, tile_extent, num_array_values):
    dims = []
    if isinstance(num_array_values, tuple):
        dims.append(tiledb.Dim(ctx, domain=(0, num_array_values[0] - 1),
                               tile=tile_extent[0], dtype=np.uint32))
        dims.append(tiledb.Dim(ctx, domain=(0, num_array_values[1] - 1),
                               tile=tile_extent[1], dtype=np.uint32))
    else:
        dims.append(tiledb.Dim(ctx, domain=(0, num_array_values - 1),
                               tile=tile_extent, dtype=np.uint32))
    tiledb_domain = tiledb.Domain(ctx, *dims)
    tiledb_schema = tiledb.ArraySchema(ctx, domain=tiledb_domain, sparse=False,
                                       attrs=[tiledb.Attr(ctx, name='a', dtype=np.float64,
                                                          compressor=('blosc-lz', 4))])
    if enable_s3:
        array_path = 's3://{}/{}'.format(s3_bucket_name, tiledb_array_name)
    else:
        array_path = tiledb_array_name
    tiledb.DenseArray.create(array_path, tiledb_schema)


def write_tiledb_array(ctx, array_data):
    if enable_s3:
        array_path = 's3://{}/{}'.format(s3_bucket_name, tiledb_array_name)
    else:
        array_path = tiledb_array_name
    with tiledb.DenseArray(ctx, array_path, mode='w') as A:
        A[:] = array_data


def write_bcolz_array(chunklen, array_data):
    A = bcolz.carray(array_data, rootdir=bcolz_array_name, mode='w', chunklen=chunklen)
    A.flush()


def write_zarr_array(chunks, array_data):
    if enable_s3:
        store = zarr_s3_store
    else:
        store = zarr_array_name
    A = zarr.open(store, mode='w', shape=array_data.shape, chunks=chunks, dtype='f8')
    A[:] = array_data


def dir_stats(dir):
    num_files, total_bytes = 0, 0
    for path, dirs, files in os.walk(dir):
        for f in files:
            num_files += 1
            total_bytes += os.path.getsize(os.path.join(path, f))
    return num_files, total_bytes / (1024 * 1024)


def bench_1d():
    # Create synthetic array data
    num_array_values = 100000000
    array_data = np.random.rand(num_array_values)
    array_mb = array_data.nbytes / (1024 * 1024.0)

    tile_extents = [1000, 10000, 100000, 1000000, 10000000]
    num_trials = 5
    use_tile_cache = False
    run_bcolz_tests = True
    run_tiledb_tests = True
    run_zarr_tests = True

    array_stats = {t_ext: {'tiledb': {}, 'bcolz': {}, 'zarr': {}} for t_ext in tile_extents}
    timings = {t_ext: {'tiledb': {}, 'bcolz': {}, 'zarr': {}} for t_ext in tile_extents}

    print('Total uncompressed array data size: {:.2f} MB.'.format(array_mb))
    for t_ext in tile_extents:
        remove_arrays()

        if run_tiledb_tests:
            drop_fs_cache()
            ctx = tiledb.Ctx(config)
            start = time.time()
            create_tiledb_array(ctx, t_ext, num_array_values)
            write_tiledb_array(ctx, array_data)
            sync_fs()
            array_stats[t_ext]['tiledb']['creation'] = time.time() - start
            array_stats[t_ext]['tiledb']['num_files'], array_stats[t_ext]['tiledb']['mb'] = dir_stats(tiledb_array_name)

            times = []
            ctx = tiledb.Ctx(config)
            for i in range(0, num_trials):
                drop_fs_cache()
                ctx_to_use = ctx if use_tile_cache else tiledb.Ctx()
                start = time.time()
                with tiledb.DenseArray(ctx_to_use, tiledb_array_name, mode='r') as A:
                    data = A[:]['a']
                times.append(time.time() - start)
            timings[t_ext]['tiledb']['read_whole_array'] = np.median(times)

            times = []
            ctx = tiledb.Ctx(config)
            for i in range(0, num_trials):
                drop_fs_cache()
                ctx_to_use = ctx if use_tile_cache else tiledb.Ctx()
                start = time.time()
                with tiledb.DenseArray(ctx_to_use, tiledb_array_name, mode='r') as A:
                    data = A[100:101]['a']
                times.append(time.time() - start)
            timings[t_ext]['tiledb']['read_1_cell'] = np.median(times)

        if run_bcolz_tests:
            drop_fs_cache()
            start = time.time()
            write_bcolz_array(t_ext, array_data)
            sync_fs()
            array_stats[t_ext]['bcolz']['creation'] = time.time() - start
            array_stats[t_ext]['bcolz']['num_files'], array_stats[t_ext]['bcolz']['mb'] = dir_stats(bcolz_array_name)

            times = []
            for i in range(0, num_trials):
                drop_fs_cache()
                start = time.time()
                with bcolz.carray(rootdir=bcolz_array_name, mode='r') as A:
                    data = A[:]
                times.append(time.time() - start)
            timings[t_ext]['bcolz']['read_whole_array'] = np.median(times)

            times = []
            for i in range(0, num_trials):
                drop_fs_cache()
                start = time.time()
                with bcolz.carray(rootdir=bcolz_array_name, mode='r') as A:
                    data = A[100:101]
                times.append(time.time() - start)
            timings[t_ext]['bcolz']['read_1_cell'] = np.median(times)

        if run_zarr_tests:
            drop_fs_cache()
            start = time.time()
            write_zarr_array(t_ext, array_data)
            sync_fs()
            array_stats[t_ext]['zarr']['creation'] = time.time() - start
            array_stats[t_ext]['zarr']['num_files'], array_stats[t_ext]['zarr']['mb'] = dir_stats(zarr_array_name)

            times = []
            for i in range(0, num_trials):
                drop_fs_cache()
                start = time.time()
                A = zarr.open(zarr_array_name, mode='r')
                data = A[:]
                times.append(time.time() - start)
            timings[t_ext]['zarr']['read_whole_array'] = np.median(times)

            times = []
            for i in range(0, num_trials):
                drop_fs_cache()
                start = time.time()
                A = zarr.open(zarr_array_name, mode='r')
                data = A[100:101]
                times.append(time.time() - start)
            timings[t_ext]['zarr']['read_1_cell'] = np.median(times)

        print('Array tile extent {}\n'.format(t_ext))
        print(('{:>10}{:>25}{:>25}{:>25}').format('', 'Creation time (sec)', 'Size (MB)', '# files'))
        row_fmt = '{:>10}{:>25.3f}{:>25.2f}{:>25}'
        for lib in ['tiledb', 'zarr', 'bcolz']:
            print(row_fmt.format(lib,
                                 array_stats[t_ext][lib]['creation'],
                                 array_stats[t_ext][lib]['mb'],
                                 array_stats[t_ext][lib]['num_files']))
        print()
        print(('{:>10}{:>25}{:>25}').format('', 'Read whole array (sec)', 'Read 1 cell (sec)'))
        row_fmt = '{:>10}{:>25.3f}{:>25.3f}'
        for lib in ['tiledb', 'zarr', 'bcolz']:
            print(row_fmt.format(lib,
                                 timings[t_ext][lib]['read_whole_array'],
                                 timings[t_ext][lib]['read_1_cell']))
        print()


def bench_2d():
    # Create synthetic array data
    num_array_values = 10000, 10000
    array_data = np.random.rand(*num_array_values)
    array_mb = array_data.nbytes / (1024 * 1024.0)

    # tile_extents = [(100, 100), (1000, 1000), (5000, 2000)]
    tile_extents = [(1000, 1000), (5000, 2000)]
    num_trials = 5
    use_tile_cache = False
    run_bcolz_tests = False
    run_tiledb_tests = True
    run_zarr_tests = True

    if enable_s3:
        tiledb_array_path = 's3://{}/{}'.format(s3_bucket_name, tiledb_array_name)
        zarr_store = zarr_s3_store
    else:
        tiledb_array_path = tiledb_array_name
        zarr_store = zarr_array_name
    array_stats = {t_ext: {'tiledb': {}, 'bcolz': {}, 'zarr': {}} for t_ext in tile_extents}
    timings = {t_ext: {'tiledb': {}, 'bcolz': {}, 'zarr': {}} for t_ext in tile_extents}

    print('Total uncompressed array data size: {:.2f} MB.'.format(array_mb))
    for t_ext in tile_extents:
        remove_arrays()

        if run_tiledb_tests:
            drop_fs_cache()
            ctx = tiledb.Ctx(config)
            start = time.time()
            create_tiledb_array(ctx, t_ext, num_array_values)
            write_tiledb_array(ctx, array_data)
            sync_fs()
            array_stats[t_ext]['tiledb']['creation'] = time.time() - start
            array_stats[t_ext]['tiledb']['num_files'], array_stats[t_ext]['tiledb']['mb'] = dir_stats(tiledb_array_name)

            times = []
            ctx = tiledb.Ctx(config)
            for i in range(0, num_trials):
                print('.', end='') ; sys.stdout.flush()
                drop_fs_cache()
                ctx_to_use = ctx if use_tile_cache else tiledb.Ctx()
                start = time.time()
                with tiledb.DenseArray(ctx_to_use, tiledb_array_path, mode='r') as A:
                    data = A[:]['a']
                times.append(time.time() - start)
            timings[t_ext]['tiledb']['read_whole_array'] = np.median(times)

            times = []
            ctx = tiledb.Ctx(config)
            for i in range(0, num_trials):
                print('.', end='') ; sys.stdout.flush()
                drop_fs_cache()
                ctx_to_use = ctx if use_tile_cache else tiledb.Ctx()
                start = time.time()
                with tiledb.DenseArray(ctx_to_use, tiledb_array_path, mode='r') as A:
                    data = A[100:101, 100:101]['a']
                times.append(time.time() - start)
            timings[t_ext]['tiledb']['read_1_cell'] = np.median(times)

            times = []
            ctx = tiledb.Ctx(config)
            for i in range(0, num_trials):
                print('.', end='') ; sys.stdout.flush()
                drop_fs_cache()
                ctx_to_use = ctx if use_tile_cache else tiledb.Ctx()
                start = time.time()
                with tiledb.DenseArray(ctx_to_use, tiledb_array_path, mode='r') as A:
                    data = A[1000:5000, 1000:5000]['a']
                times.append(time.time() - start)
            timings[t_ext]['tiledb']['read_subarray'] = np.median(times)

        if run_bcolz_tests:
            drop_fs_cache()
            start = time.time()
            try:
                write_bcolz_array(t_ext[0] * t_ext[1], array_data)
                success = True
            except:
                success = False
                array_stats[t_ext]['bcolz']['creation'] = time.time() - start
                array_stats[t_ext]['bcolz']['num_files'], array_stats[t_ext]['bcolz']['mb'] = float('nan'), float('nan')
                timings[t_ext]['bcolz']['read_whole_array'] = float('nan')
                timings[t_ext]['bcolz']['read_1_cell'] = float('nan')
                timings[t_ext]['bcolz']['read_subarray'] = float('nan')
            sync_fs()

            if success:
                array_stats[t_ext]['bcolz']['creation'] = time.time() - start
                array_stats[t_ext]['bcolz']['num_files'], array_stats[t_ext]['bcolz']['mb'] = dir_stats(
                    bcolz_array_name)

                times = []
                for i in range(0, num_trials):
                    drop_fs_cache()
                    start = time.time()
                    with bcolz.carray(rootdir=bcolz_array_name, mode='r') as A:
                        data = A[:]
                    times.append(time.time() - start)
                timings[t_ext]['bcolz']['read_whole_array'] = np.median(times)

                times = []
                for i in range(0, num_trials):
                    drop_fs_cache()
                    start = time.time()
                    with bcolz.carray(rootdir=bcolz_array_name, mode='r') as A:
                        data = A[100:101, 100:101]
                    times.append(time.time() - start)
                timings[t_ext]['bcolz']['read_1_cell'] = np.median(times)

                times = []
                for i in range(0, num_trials):
                    drop_fs_cache()
                    start = time.time()
                    with bcolz.carray(rootdir=bcolz_array_name, mode='r') as A:
                        data = A[1000:5000, 1000:5000]
                    times.append(time.time() - start)
                timings[t_ext]['bcolz']['read_subarray'] = np.median(times)

        if run_zarr_tests:
            drop_fs_cache()
            start = time.time()
            write_zarr_array(t_ext, array_data)
            sync_fs()
            array_stats[t_ext]['zarr']['creation'] = time.time() - start
            array_stats[t_ext]['zarr']['num_files'], array_stats[t_ext]['zarr']['mb'] = dir_stats(zarr_array_name)

            times = []
            for i in range(0, num_trials):
                print('.', end='') ; sys.stdout.flush()
                drop_fs_cache()
                start = time.time()
                A = zarr.open(zarr_store, mode='r')
                data = A[:]
                times.append(time.time() - start)
            timings[t_ext]['zarr']['read_whole_array'] = np.median(times)

            times = []
            for i in range(0, num_trials):
                print('.', end='') ; sys.stdout.flush()
                drop_fs_cache()
                start = time.time()
                A = zarr.open(zarr_store, mode='r')
                data = A[100:101, 100:101]
                times.append(time.time() - start)
            timings[t_ext]['zarr']['read_1_cell'] = np.median(times)

            times = []
            for i in range(0, num_trials):
                print('.', end='') ; sys.stdout.flush()
                drop_fs_cache()
                start = time.time()
                A = zarr.open(zarr_store, mode='r')
                data = A[1000:5000, 1000:5000]
                times.append(time.time() - start)
            timings[t_ext]['zarr']['read_subarray'] = np.median(times)

        print('Array tile extent {}'.format(t_ext))
        print(('{:>10}{:>25}{:>25}{:>25}').format('', 'Creation time (sec)', 'Size (MB)', '# files'))
        row_fmt = '{:>10}{:>25.3f}{:>25.2f}{:>25}'
        for lib in ['tiledb', 'zarr']:
            print(row_fmt.format(lib,
                                 array_stats[t_ext][lib]['creation'],
                                 array_stats[t_ext][lib]['mb'],
                                 array_stats[t_ext][lib]['num_files']))

        print()
        print(('{:>10}{:>25}{:>25}{:>25}').format('', 'Read whole array (sec)', 'Read 1 cell (sec)',
                                                  'Slice 1k:5k,1k:5k (sec)'))
        row_fmt = '{:>10}{:>25.3f}{:>25.3f}{:>25.3f}'
        for lib in ['tiledb', 'zarr']:
            print(row_fmt.format(lib,
                                 timings[t_ext][lib]['read_whole_array'],
                                 timings[t_ext][lib]['read_1_cell'],
                                 timings[t_ext][lib]['read_subarray']))
        print()


# bench_1d()
bench_2d()