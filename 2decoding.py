
'''
========================================================================
FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
We create a combination of the t-maps and the anatomical mask.
To this end, we multiply the anatomical mask with each t-map.
As the anatomical conists of binary values (zeros and ones) a
multiplication results in t-map masked by the anatomical ROI.
========================================================================
'''
#v= tmaps_masked[0]
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(v[:,0],v[:,1],v[:,2], zdir='z', c= 'red')
#plt.show()

# check if any value in the supposedly binary mask is bigger than 1:
if np.any(mask_vis > 1):
    logging.info('WARNING: detected values > 1 in the anatomical ROI!')
    sys.exit("Values > 1 in the anatomical ROI!")
# get combination of anatomical mask and t-map
tmaps_masked = [np.multiply(mask_vis, i) for i in copy.deepcopy(tmaps)]
# masked tmap into image like object:
tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap, copy.deepcopy(tmaps_masked))]

for i, path in enumerate(tmaps_masked_img):
    path_save = opj(path_out_masks, '%s_run-%02d_tmap_masked.nii.gz' % (sub, i + 1))
    path.to_filename(path_save)

# plot masked t-maps
logging.info('plotting masked tmaps with anatomical as background:')
for i, path in enumerate(tmaps_masked_img):
    logging.info('plotting masked tmap %d of %d' % (i+1, len(tmaps_masked_img)))
    path_save = opj(path_out_figs, '%s_run-%02d_tmap_masked.png' % (sub, i+1))
    plotting.plot_roi(path, anat, title=os.path.basename(path_save),
                      output_file=path_save, colorbar=True)
'''
========================================================================
FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
We threshold the masked t-maps, selecting only voxels above AND
below this threshold. We then extract these data and count how
many voxels were above and / or below the threshold in total.
========================================================================
'''
# set the threshold:
threshold = params['mri']['thresh']
logging.info('thresholding t-maps with a threshold of %s' % str(threshold))
# threshold the masked tmap image:
tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]

logging.info('plotting thresholded tmaps with anatomical as background:')
for i, path in enumerate(tmaps_masked_thresh_img):
    path_save = opj(path_out_figs, '%s_run-%02d_tmap_masked_thresh.png' % (sub, i+1))
    logging.info('plotting masked tmap %s (%d of %d)'
                 % (path_save, i + 1, len(tmaps_masked_thresh_img)))
    plotting.plot_roi(path, anat, title=os.path.basename(path_save),
                      output_file=path_save, colorbar=True)

# extract data from the thresholded images
tmaps_masked_thresh = [image.load_img(i).get_data().astype(float) for i in tmaps_masked_thresh_img]

# calculate the number of tmap voxels:
num_tmap_voxel = [np.size(i) for i in copy.deepcopy(tmaps_masked_thresh)]
logging.info('number of feature selected voxels: %s' % pformat(num_tmap_voxel))

num_above_voxel = [np.count_nonzero(i) for i in copy.deepcopy(tmaps_masked_thresh)]
logging.info('number of voxels above threshold: %s' % pformat(num_above_voxel))

num_below_voxel = [np.count_nonzero(i == 0) for i in copy.deepcopy(tmaps_masked_thresh)]
logging.info('number of voxels below threshold: %s' % pformat(num_below_voxel))

# plot the distribution of t-values:
for i, run_mask in enumerate(tmaps_masked_thresh):
    masked_tmap_flat = run_mask.flatten()
    masked_tmap_flat = masked_tmap_flat[~np.isnan(masked_tmap_flat)]
    masked_tmap_flat = masked_tmap_flat[~np.isnan(masked_tmap_flat) & ~(masked_tmap_flat == 0)]
    path_save = opj(path_out_figs, '%s_run-%02d_tvalue_distribution.png' % (sub, i+1))
    logging.info('plotting thresholded t-value distribution %s (%d of %d)'
                 % (path_save, i+1, len(tmaps_masked_thresh)))
    fig = plt.figure()
    plt.hist(masked_tmap_flat, bins='auto')
    plt.xlabel('t-values')
    plt.ylabel('number')
    plt.title('t-value distribution (%s, run-%02d)' % (sub, i+1))
    plt.savefig(path_save)

# create a dataframe with the number of voxels
df_thresh = pd.DataFrame({
    'id': [sub] * n_run,
    'run': np.arange(1,n_run+1),
    'n_total': num_tmap_voxel,
    'n_above': num_above_voxel,
    'n_below': num_below_voxel
})
file_name = opj(path_out_data, '%s_thresholding.csv' % (sub))
df_thresh.to_csv(file_name, sep=',', index=False)

'''
========================================================================
FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS
We set all voxels above and below the threshold to 1 and all voxels
that were not selected to 0.
========================================================================
'''
# replace all NaNs with 0:
tmaps_masked_thresh_bin = [np.where(np.isnan(i), 0, i) for i in copy.deepcopy(tmaps_masked_thresh)]
# replace all other values with 1:
tmaps_masked_thresh_bin = [np.where(i > 0, 1, i) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
# turn the 3D-array into booleans:
tmaps_masked_thresh_bin = [i.astype(bool) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
# create image like object:
masks_final = [image.new_img_like(path_func_task[0], i.astype(np.int)) for i in copy.deepcopy(tmaps_masked_thresh_bin)]

logging.info('plotting final masks with anatomical as background:')
for i, path in enumerate(masks_final):
    filename = '%s_run-%02d_tmap_masked_thresh.nii.gz'\
               % (sub, i + 1)
    path_save = opj(path_out_masks, filename)
    logging.info('saving final mask %s (%d of %d)'
                 % (path_save, i+1, len(masks_final)))
    path.to_filename(path_save)
    path_save = opj(path_out_figs, '%s_run-%02d_visual_final_mask.png'
                    % (sub, i + 1))
    logging.info('plotting final mask %s (%d of %d)'
                 % (path_save, i + 1, len(masks_final)))
    plotting.plot_roi(path, anat, title=os.path.basename(path_save),
                      output_file=path_save, colorbar=True)
'''
========================================================================
LOAD SMOOTHED FMRI DATA FOR ALL FUNCTIONAL TASK RUNS:
========================================================================
'''
# load smoothed functional mri data for all eight task runs:
logging.info('loading %d functional task runs ...' % len(path_func_task))
data_task = [image.load_img(i) for i in path_func_task]
logging.info('loading successful!')