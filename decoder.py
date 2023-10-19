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
'''
========================================================================
DEFINE THE CLASSIFIERS
========================================================================
'''
class_labels = ['cat', 'chair', 'face', 'house', 'shoe']
# create a dictionary with all values as independent instances:
# see here: https://bit.ly/2J1DvZm
clf_set = {key: LogisticRegression(
    C=1., penalty='l2', multi_class='ovr', solver='lbfgs',
    max_iter=4000, class_weight='balanced', random_state=42) for key in class_labels}
classifiers = {
    'log_reg': LogisticRegression(
        C=1., penalty='l2', multi_class='multinomial', solver='lbfgs',
        max_iter=4000, class_weight='balanced', random_state=42)}
clf_set.update(classifiers)
'''
========================================================================
1. SPLIT THE EVENTS DATAFRAME FOR EACH TASK CONDITION
2. RESET THE INDICES OF THE DATAFRAMES
3. SORT THE ROWS OF ALL DATAFRAMES IN CHRONOLOGICAL ORDER
4. PRINT THE NUMBER OF TRIALS OF EACH TASK CONDITION
========================================================================
'''


class TaskData:
    """

    """
    def __init__(
            self, events, condition, name, trial_type, bold_delay=0,
            interval=1, t_tr=1.25, num_vol_run=530):
        import pandas as pd
        # define name of the task data subset:
        self.name = name
        # define the task condition the task data subset if from:
        self.condition = condition
        # define the delay (in seconds) by which onsets are moved:
        self.bold_delay = bold_delay
        # define the number of TRs from event onset that should be selected:
        self.interval = interval
        # define the repetition time (TR) of the mri data acquisition:
        self.t_tr = t_tr
        # define the number of volumes per task run:
        self.num_vol_run = num_vol_run
        # select events: upright stimulus, correct answer trials only:
        if trial_type == 'stimulus':
            self.events = events.loc[
                          (events['condition'] == condition) &
                          (events['trial_type'] == trial_type) &
                          (events['stim_orient'] == 0) &
                          (events['serial_position'] == 1) &
                          (events['accuracy'] != 0),
                          :]
        elif trial_type == 'cue':
            self.events = events.loc[
                          (events['condition'] == condition) &
                          (events['trial_type'] == trial_type),
                          :]
        # reset the indices of the data frame:
        self.events.reset_index()
        # sort all values by session and run:
        self.events.sort_values(by=['session', 'run_session'])
        # call further function upon initialization:
        self.define_trs()
        self.get_stats()

    def define_trs(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events['onset']
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = self.events['onset'] + self.bold_delay
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        # add the number of run volumes to the tr indices:
        run_volumes = (self.events['run_study']-1) * self.num_vol_run
        # add the number of volumes of each run:
        trs = round(self.peak_trs + run_volumes)
        # copy the relevant trs as often as specified by the interval:
        a = np.transpose(np.tile(trs, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        b = np.full((len(trs), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(a, b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes)

    def get_stats(self):
        import numpy as np
        self.num_trials = len(self.events)
        self.runs = np.repeat(np.array(self.events['run_study'], dtype=int), self.interval)
        self.trials = np.repeat(np.array(self.events['trial'], dtype=int), self.interval)
        self.sess = np.repeat(np.array(self.events['session'], dtype=int), self.interval)
        self.stim = np.repeat(np.array(self.events['stim_label'], dtype=object), self.interval)
        self.itis = np.repeat(np.array(self.events['interval_time'], dtype=float), self.interval)
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)

    def zscore(self, signals, run_list, t_tr=1.25):
        from nilearn.signal import clean
        import numpy as np
        # get boolean indices for all run indices in the run list:
        run_indices = np.isin(self.runs, list(run_list))
        # standardize data all runs in the run list:
        self.data_zscored = clean(
            signals=signals[self.trs[run_indices]],
            sessions=self.runs[run_indices],
            t_r=t_tr,
            detrend=False,
            standardize=True)

    def predict(self, clf, run_list):
        # import packages:
        import pandas as pd
        # get classifier class predictions:
        pred_class = clf.predict(self.data_zscored)
        # get classifier probabilistic predictions:
        pred_proba = clf.predict_proba(self.data_zscored)
        # get the classes of the classifier:
        classes_names = clf.classes_
        # get boolean indices for all run indices in the run list:
        run_indices = np.isin(self.runs, list(run_list))
        # create a dataframe with the probabilities of each class:
        df = pd.DataFrame(pred_proba, columns=classes_names)
        # get the number of predictions made:
        num_pred = len(df)
        # get the number of trials in the test set:
        num_trials = int(num_pred / self.interval)
        # add the predicted class label to the dataframe:
        df['pred_label'] = pred_class
        # add the true stimulus label to the dataframe:
        df['stim'] = self.stim[run_indices]
        # add the volume number (TR) to the dataframe:
        df['tr'] = self.trs[run_indices]
        # add the sequential TR to the dataframe:
        df['seq_tr'] = np.tile(np.arange(1, self.interval + 1), num_trials)
        # add the counter of trs on which the stimulus was presented
        df['stim_tr'] = self.stim_trs[run_indices]
        # add the trial number to the dataframe:
        df['trial'] = self.trials[run_indices]
        # add the run number to the dataframe:
        df['run_study'] = self.runs[run_indices]
        # add the session number to the dataframe:
        df['session'] = self.sess[run_indices]
        # add the inter trial interval to the dataframe:
        df['tITI'] = self.itis[run_indices]
        # add the participant id to the dataframe:
        df['id'] = np.repeat(self.events['subject'].unique(), num_pred)
        # add the name of the classifier to the dataframe:
        df['test_set'] = np.repeat(self.name, num_pred)
        # return the dataframe:
        return df


train_odd_peak = TaskData(
        events=df_events, condition='oddball', trial_type='stimulus',
        bold_delay=4, interval=1, name='train-odd_peak')
test_odd_peak = TaskData(
        events=df_events, condition='oddball', trial_type='stimulus',
        bold_delay=4, interval=1, name='test-odd_peak')
test_odd_long = TaskData(
        events=df_events, condition='oddball', trial_type='stimulus',
        bold_delay=0, interval=7, name='test-odd_long')
test_seq_long = TaskData(
        events=df_events, condition='sequence', trial_type='stimulus',
        bold_delay=0, interval=13, name='test-seq_long')
test_rep_long = TaskData(
        events=df_events, condition='repetition', trial_type='stimulus',
        bold_delay=0, interval=13, name='test-rep_long')
test_seq_cue = TaskData(
        events=df_events, condition='sequence', trial_type='cue',
        bold_delay=0, interval=5, name='test-seq_cue')
test_rep_cue = TaskData(
        events=df_events, condition='repetition', trial_type='cue',
        bold_delay=0, interval=5, name='test-rep_cue')
test_sets = [
        test_odd_peak, test_odd_long, test_seq_long, test_rep_long,
        test_seq_cue, test_rep_cue
        ]
'''
========================================================================
SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF ALL TASK CONDITIONS
Standardize features by removing the mean and scaling to unit variance.
Centering and scaling happen independently on each feature by computing
the relevant statistics on the samples in the training set. Here,
we standardize the features of all trials run-wise but separated for
each task condition (oddball, sequence, and repetition condition).
========================================================================
'''


def detrend(data, t_tr=1.25):
    from nilearn.signal import clean
    data_detrend = clean(
        signals=data, t_r=t_tr, detrend=True, standardize=False)
    return data_detrend


def show_weights(array):
    # https://stackoverflow.com/a/50154388
    import numpy as np
    import seaborn as sns
    n_samples = array.shape[0]
    classes, bins = np.unique(array, return_counts=True)
    n_classes = len(classes)
    weights = n_samples / (n_classes * bins)
    sns.barplot(classes, weights)
    plt.xlabel('class label')
    plt.ylabel('weight')
    plt.show()


def melt_df(df, melt_columns):
    # save the column names of the dataframe in a list:
    column_names = df.columns.tolist()
    # remove the stimulus classes from the column names;
    id_vars = [x for x in column_names if x not in melt_columns]
    # melt the dataframe creating one column with value_name and var_name:
    df_melt = pd.melt(
            df, value_name='probability', var_name='class', id_vars=id_vars)
    # return the melted dataframe:
    return df_melt


data_list = []
runs = list(range(1, n_run+1))
#mask_label = 'cv'

logging.info('starting leave-one-run-out cross-validation')
for mask_label in ['cv', 'cv_hpc']:
    logging.info('testing in mask %s' % (mask_label))
    for run in runs:
        logging.info('testing on run %d of %d ...' % (run, len(runs)))
        # define the run indices for the training and test set:
        train_runs = [x for x in runs if x != run]
        test_runs = [x for x in runs if x == run]
        # get the feature selection mask of the current run:
        if mask_label == 'cv':
            mask_run = masks_final[runs.index(run)]
        elif mask_label == 'cv_hpc':
            mask_run = path_mask_hpc_task[runs.index(run)]
        # extract smoothed fMRI data from the mask for the cross-validation fold:
        masked_data = [masking.apply_mask(i, mask_run) for i in data_task]
        # detrend the masked fMRI data separately for each run:
        data_detrend = [detrend(i) for i in masked_data]
        # combine the detrended data of all runs:
        data_detrend = np.vstack(data_detrend)
        # loop through all classifiers in the classifier set:
        for clf_name, clf in clf_set.items():
            # print classifier:
            logging.info('classifier: %s' % clf_name)
            # fit the classifier to the training data:
            train_odd_peak.zscore(signals=data_detrend, run_list=train_runs)
            # get the example labels:
            train_stim = copy.deepcopy(train_odd_peak.stim[train_odd_peak.runs != run])
            # replace labels for single-label classifiers:
            if clf_name in class_labels:
                # replace all other labels with other
                train_stim = ['other' if x != clf_name else x for x in train_stim]
                # turn into a numpy array
                train_stim = np.array(train_stim, dtype=object)
            # check weights:
            #show_weights(array=train_stim)
            # train the classifier
            clf.fit(train_odd_peak.data_zscored, train_stim)
            # classifier prediction: predict on test data and save the data:
            for test_set in test_sets:
                logging.info('testing on test set %s' % test_set.name)
                test_set.zscore(signals=data_detrend, run_list=test_runs)
                if test_set.data_zscored.size < 0:
                    continue
                # create dataframe containing classifier predictions:
                df_pred = test_set.predict(clf=clf, run_list=test_runs)
                # add the current classifier as a new column:
                df_pred['classifier'] = np.repeat(clf_name, len(df_pred))
                # add a label that indicates the mask / training regime:
                df_pred['mask'] = np.repeat(mask_label, len(df_pred))
                # melt the data frame:
                df_pred_melt = melt_df(df=df_pred, melt_columns=train_stim)
                # append dataframe to list of dataframe results:
                data_list.append(df_pred_melt)
