
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
'''
========================================================================
DECODING ON RESTING STATE DATA:
========================================================================
'''
logging.info('Loading fMRI data of %d resting state runs ...' % len(path_rest))
data_rest = [image.load_img(i) for i in path_rest]
logging.info('loading successful!')

# combine all masks from the feature selection by intersection:
def multimultiply(arrays):
    import copy
    # start with the first array:
    array_union = copy.deepcopy(arrays[0].astype(np.int))
    # loop through all arrays
    for i in range(len(arrays)):
        # multiply every array with all previous array
        array_union = np.multiply(array_union, copy.deepcopy(arrays[i].astype(np.int)))
    # return the union of all arrays:
    return(array_union)


for mask_label in ['union', 'union_hpc']:
    # calculate the union of all masks by multiplication:
    if mask_label == 'union':
        masks_union = multimultiply(tmaps_masked_thresh_bin).astype(int).astype(bool)
    elif mask_label == 'union_hpc':
        masks_union = multimultiply(mask_hpc).astype(int).astype(bool)
    # masked tmap into image like object:
    masks_union_nii = image.new_img_like(path_func_task[0], masks_union)
    path_save = opj(path_out_masks, '{}_task-rest_mask-{}.nii.gz'.format(sub, mask_label))
    masks_union_nii.to_filename(path_save)
    # mask all resting state runs with the averaged feature selection masks:
    data_rest_masked = [masking.apply_mask(i, masks_union_nii) for i in data_rest]
    # detrend and standardize each resting state run separately:
    data_rest_final = [clean(i, detrend=True, standardize=True) for i in data_rest_masked]
    # mask all functional task runs separately:
    data_task_masked = [masking.apply_mask(i, masks_union_nii) for i in data_task]
    # detrend each task run separately:
    data_task_masked_detrend = [clean(i, detrend=True, standardize=False) for i in data_task_masked]
    # combine the detrended data of all runs:
    data_task_masked_detrend = np.vstack(data_task_masked_detrend)
    # select only oddball data and standardize:
    train_odd_peak.zscore(signals=data_task_masked_detrend, run_list=runs)
    # write session and run labels:
    ses_labels = [i.split(sub + "_")[1].split("_task")[0] for i in path_rest]
    run_labels = [i.split("prenorm_")[1].split("_space")[0] for i in path_rest]
    file_names = ['_'.join([a, b]) for (a, b) in zip(ses_labels, run_labels)]
    rest_interval = 1
    # save the voxel patterns:
    num_voxels = len(train_odd_peak.data_zscored[0])
    voxel_labels = ['v' + str(x) for x in range(num_voxels)]
    df_patterns = pd.DataFrame(train_odd_peak.data_zscored, columns=voxel_labels)
    # add the stimulus labels to the dataframe:
    df_patterns['label'] = copy.deepcopy(train_odd_peak.stim)
    # add the participant id to the dataframe:
    df_patterns['id'] = np.repeat(df_events['subject'].unique(), len(train_odd_peak.stim))
    # add the mask label:
    df_patterns['mask'] = np.repeat(mask_label, len(train_odd_peak.stim))
    # split pattern dataframe by label:
    df_pattern_list = [df_patterns[df_patterns['label'] == i] for i in df_patterns['label'].unique()]
    # create file path to save the dataframe:
    file_paths = [opj(path_out_data, '{}_voxel_patterns_{}_{}'.format(sub, mask_label, i)) for i in df_patterns['label'].unique()]
    # save the final dataframe to a .csv-file:
    [i.to_csv(j + '.csv', sep=',', index=False) for (i, j) in zip(df_pattern_list, file_paths)]
    # save only the voxel patterns as niimg-like objects
    [masking.unmask(X=i.loc[:, voxel_labels].to_numpy(), mask_img=masks_union_nii).to_filename(j + '.nii.gz') for (i, j) in zip(df_pattern_list, file_paths)]
    #[image.new_img_like(path_func_task[0], i.loc[:, voxel_labels].to_numpy()).to_filename(j + '.nii.gz') for (i, j) in zip(df_pattern_list, file_paths)]
    # decoding on resting state data:
    for clf_name, clf in clf_set.items():
        # print classifier name:
        logging.info('classifier: %s' % clf_name)
        # get the example labels for all slow trials:
        train_stim = copy.deepcopy(train_odd_peak.stim)
        # replace labels for single-label classifiers:
        if clf_name in class_labels:
            # replace all other labels with other
            train_stim = ['other' if x != clf_name else x for x in train_stim]
            # turn into a numpy array
            train_stim = np.array(train_stim, dtype=object)
        # train the classifier
        clf.fit(train_odd_peak.data_zscored, train_stim)
        # classifier prediction: predict on test data and save the data:
        pred_rest_class = [clf.predict(i) for i in data_rest_final]
        pred_rest_proba = [clf.predict_proba(i) for i in data_rest_final]
        # get the class names of the classifier:
        classes_names = clf.classes_
        # save classifier predictions on resting state scans
        for t, name in enumerate(pred_rest_proba):
            # create a dataframe with the probabilities of each class:
            df_rest_pred = pd.DataFrame(
                    pred_rest_proba[t], columns=classes_names)
            # get the number of predictions made:
            num_pred = len(df_rest_pred)
            # get the number of trials in the test set:
            num_trials = int(num_pred / rest_interval)
            # add the predicted class label to the dataframe:
            df_rest_pred['pred_label'] = pred_rest_class[t]
            # add the true stimulus label to the dataframe:
            df_rest_pred['stim'] = np.full(num_pred, np.nan)
            # add the volume number (TR) to the dataframe:
            df_rest_pred['tr'] = np.arange(1, num_pred + 1)
            # add the sequential TR to the dataframe:
            df_rest_pred['seq_tr'] = np.arange(1, num_pred + 1)
            # add the trial number to the dataframe:
            df_rest_pred['trial'] = np.tile(
                    np.arange(1, rest_interval + 1), num_trials)
            # add the run number to the dataframe:
            df_rest_pred['run_study'] = np.repeat(run_labels[t], num_pred)
            # add the session number to the dataframe:
            df_rest_pred['session'] = np.repeat(ses_labels[t], len(df_rest_pred))
            # add the inter trial interval to the dataframe:
            df_rest_pred['tITI'] = np.tile('rest', num_pred)
            # add the participant id to the dataframe:
            df_rest_pred['id'] = np.repeat(df_events['subject'].unique(), num_pred)
            # add the name of the classifier to the dataframe:
            df_rest_pred['test_set'] = np.repeat('rest', num_pred)
            # add the name of the classifier to the dataframe;
            df_rest_pred['classifier'] = np.repeat(clf_name, num_pred)
            # add a label that indicates the mask / training regime:
            df_rest_pred['mask'] = np.repeat(mask_label, len(df_rest_pred))
            # melt the data frame:
            df_pred_melt = melt_df(df=df_rest_pred, melt_columns=train_stim)
            # append dataframe to list of dataframe results:
            data_list.append(df_pred_melt)
        # run classifier trained on all runs across all test sets:
        for test_set in test_sets:
            logging.info('testing on test set %s' % test_set.name)
            test_set.zscore(signals=data_task_masked_detrend, run_list=runs)
            if test_set.data_zscored.size < 0:
                continue
            # create dataframe containing classifier predictions:
            df_pred = test_set.predict(clf=clf, run_list=runs)
            # add the current classifier as a new column:
            df_pred['classifier'] = np.repeat(clf_name, len(df_pred))
            # add a label that indicates the mask / training regime:
            df_pred['mask'] = np.repeat(mask_label, len(df_pred))
            # melt the data frame:
            df_pred_melt = melt_df(df=df_pred, melt_columns=train_stim)
            # append dataframe to list of dataframe results:
            data_list.append(df_pred_melt)

# concatenate all decoding dataframes in one final dataframe:
df_all = pd.concat(data_list, sort=False)
# create file path to save the dataframe:
file_path = opj(path_out_data, '{}_decoding.csv'.format(sub))
# save the final dataframe to a .csv-file:
df_all.to_csv(file_path, sep=',', index=False)

'''
========================================================================
STOP LOGGING:
========================================================================
'''
end = time.time()
total_time = (end - start)/60
logging.info('total running time: %0.2f minutes' % total_time)
logging.shutdown()
