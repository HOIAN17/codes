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
