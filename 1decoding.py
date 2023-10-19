import glob
import os
import yaml
import logging
import time
from os.path import join as opj
import sys
import copy
from pprint import pformat
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn import plotting, image, masking
import pandas as pd
from matplotlib import pyplot as plt
import warnings
from nilearn.signal import clean
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from collections import Counter
import datalad.api as dl
'''
========================================================================
DEAL WITH ERRORS, WARNING ETC.
========================================================================
'''
warnings.filterwarnings('ignore', message='numpy.dtype size changed*')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed*')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
'''
========================================================================
SETUP NECESSARY PATHS ETC:
========================================================================
'''
# get start time of the script:
start = time.time()
# name of the current project:
project = 'highspeed'
# initialize empty variables:
sub = None
path_tardis = None
path_server = None
path_local = None
# define paths depending on the operating system (OS) platform:
if 'darwin' in sys.platform:
    # define the path to the cluster:
    path_tardis = opj(os.environ['HOME'], 'Volumes', 'tardis_beegfs', project)
    # define the path to the server:
    path_server = opj('/Volumes', 'MPRG-Neurocode', 'Data', project)
    # define the path to the local computer:
    path_local = opj(os.environ['HOME'], project, '*', '*')
    # define the subject id:
    sub = 'sub-14'
elif 'linux' in sys.platform:
    # path to the project root:
    project_name = 'highspeed-decoding'
    path_root = os.getenv('PWD').split(project_name)[0] + project_name
    # define the path to the cluster:
    path_tardis = path_root
    # define the path to the server:
    path_server = path_tardis
    # define the path to the local computer:
    path_local = opj(path_tardis, 'code', 'decoding')
    # define the subject id:
    sub = 'sub-%s' % sys.argv[1]
'''
========================================================================
LOAD PROJECT PARAMETERS:
========================================================================
'''
path_params = glob.glob(opj(path_local, '*parameters.yaml'))[0]
with open(path_params, 'rb') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
f.close()
'''
========================================================================
CREATE PATHS TO OUTPUT DIRECTORIES:
========================================================================
'''
path_decoding = opj(path_tardis, 'decoding')
path_out = opj(path_decoding, sub)
path_out_figs = opj(path_out, 'plots')
path_out_data = opj(path_out, 'data')
path_out_logs = opj(path_out, 'logs')
path_out_masks = opj(path_out, 'masks')
'''
========================================================================
CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
========================================================================
'''
for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks]:
    if not os.path.exists(path):
        os.makedirs(path)
'''
========================================================================
SETUP LOGGING:
========================================================================
'''
# remove all handlers associated with the root logger object:
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# get current data and time as a string:
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
# create path for the logging file
log = opj(path_out_logs, '%s-%s.log' % (timestr, sub))
# start logging:
logging.basicConfig(
    filename=log, level=logging.DEBUG, format='%(asctime)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S')
'''
========================================================================
DEFINE DECODING SPECIFIC PARAMETERS:
========================================================================
'''
# define the mask to be used:
mask = 'visual'  # visual or whole
# applied time-shift to account for the BOLD delay, in seconds:
bold_delay = 4  # 4, 5 or 6 secs
# define the degree of smoothing of the functional data
smooth = 4
'''
========================================================================
ADD BASIC SCRIPT INFORMATION TO THE LOGGER:
========================================================================
'''
logging.info('Running decoding script')
logging.info('operating system: %s' % sys.platform)
logging.info('project name: %s' % project)
logging.info('participant: %s' % sub)
logging.info('mask: %s' % mask)
logging.info('bold delay: %d secs' % bold_delay)
logging.info('smoothing kernel: %d mm' % smooth)
'''
========================================================================
DEFINE RELEVANT VARIABLES:
========================================================================
'''
# time of repetition (TR), in seconds:
t_tr = params['mri']['tr']
# number of volumes (TRs) for each functional task run:
n_tr_run = 530
# acquisition time window of one sequence trial, in seconds:
t_win = 16
# number of measurements that are considered per sequence time window:
n_tr_win = round(t_win / t_tr)
# number of oddball trials in the experiment:
n_tr_odd = 600
# number of sequence trials in the experiment:
n_tr_seq = 75
# number of repetition trials in the experiment:
n_tr_rep = 45
# number of scanner triggers before the experiment starts:
n_tr_wait = params['mri']['num_trigger']
# number of functional task runs in total:
n_run = params['mri']['num_runs']
# number of experimental sessions in total:
n_ses = params['mri']['num_sessions']
# number of functional task runs per session:
n_run_ses = int(n_run / n_ses)
'''
========================================================================
LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT):
========================================================================
Load the events files for the subject. The 'oddball' data serves as the
training set of the classifier. The 'sequence' data constitutes the
test set.
'''
# paths to all events files of the current subject:
path_events = opj(path_tardis, 'bids', sub, 'ses-*', 'func', '*tsv')
dl.get(glob.glob(path_events))
path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))
logging.info('found %d event files' % len(path_events))
logging.info('paths to events files (sorted):\n%s' % pformat(path_events))
# import events file and save data in dataframe:
df_events = pd.concat((pd.read_csv(f, sep='\t') for f in path_events),
                      ignore_index=True)
'''
========================================================================
CREATE PATHS TO THE MRI DATA:
========================================================================
'''
# define path to input directories:
path_fmriprep = opj(path_tardis, 'fmriprep', 'fmriprep', sub)
path_level1 = opj(path_tardis, 'glm', 'l1pipeline')
path_masks = opj(path_tardis, 'masks', 'masks')

logging.info('path to fmriprep files: %s' % path_fmriprep)
logging.info('path to level 1 files: %s' % path_level1)
logging.info('path to mask files: %s' % path_masks)
paths = {
    'fmriprep': opj(path_tardis, 'fmriprep', 'fmriprep', sub),
    'level1': opj(path_tardis, 'glm', 'l1pipeline'),
    'masks': opj(path_tardis, 'masks', 'masks')
}

# load the visual mask task files:
path_mask_vis_task = opj(path_masks, 'mask_visual', sub, '*', '*task-highspeed*.nii.gz')
path_mask_vis_task = sorted(glob.glob(path_mask_vis_task), key=lambda f: os.path.basename(f))
logging.info('found %d visual mask task files' % len(path_mask_vis_task))
logging.info('paths to visual mask task files:\n%s' % pformat(path_mask_vis_task))
dl.get(path_mask_vis_task)

# load the hippocampus mask task files:
path_mask_hpc_task = opj(path_masks, 'mask_hippocampus', sub, '*', '*task-highspeed*.nii.gz')
path_mask_hpc_task = sorted(glob.glob(path_mask_hpc_task), key=lambda f: os.path.basename(f))
logging.info('found %d hpc mask files' % len(path_mask_hpc_task))
logging.info('paths to hpc mask task files:\n%s' % pformat(path_mask_hpc_task))
dl.get(path_mask_hpc_task)

# load the whole brain mask files:
path_mask_whole_task = opj(path_fmriprep, '*', 'func', '*task-highspeed*T1w*brain_mask.nii.gz')
path_mask_whole_task = sorted(glob.glob(path_mask_whole_task), key=lambda f: os.path.basename(f))
logging.info('found %d whole-brain masks' % len(path_mask_whole_task))
logging.info('paths to whole-brain mask files:\n%s' % pformat(path_mask_whole_task))
dl.get(path_mask_whole_task)

# load the functional mri task files:
path_func_task = opj(path_level1, 'smooth', sub, '*', '*task-highspeed*nii.gz')
path_func_task = sorted(glob.glob(path_func_task), key=lambda f: os.path.basename(f))
logging.info('found %d functional mri task files' % len(path_func_task))
logging.info('paths to functional mri task files:\n%s' % pformat(path_func_task))
dl.get(path_func_task)

# define path to the functional resting state runs:
path_rest = opj(path_tardis, 'masks', 'masks', 'smooth', sub, '*', '*task-rest*nii.gz')
path_rest = sorted(glob.glob(path_rest), key=lambda f: os.path.basename(f))
logging.info('found %d functional mri rest files' % len(path_rest))
logging.info('paths to functional mri rest files:\n%s' % pformat(path_rest))
dl.get(path_rest)

# load the anatomical mri file:
path_anat = opj(path_fmriprep, 'anat', '%s_desc-preproc_T1w.nii.gz' % sub)
path_anat = sorted(glob.glob(path_anat), key=lambda f: os.path.basename(f))
logging.info('found %d anatomical mri file' % len(path_anat))
logging.info('paths to anatoimical mri files:\n%s' % pformat(path_anat))
dl.get(path_anat)

# load the confounds files:
path_confs_task = opj(path_fmriprep, '*', 'func', '*task-highspeed*confounds_regressors.tsv')
path_confs_task = sorted(glob.glob(path_confs_task), key=lambda f: os.path.basename(f))
logging.info('found %d confounds files' % len(path_confs_task))
logging.info('found %d confounds files' % len(path_confs_task))
logging.info('paths to confounds files:\n%s' % pformat(path_confs_task))
dl.get(path_confs_task)

# load the spm.mat files:
path_spm_mat = opj(path_level1, 'contrasts', sub, '*', 'SPM.mat')
path_spm_mat = sorted(glob.glob(path_spm_mat), key=lambda f: os.path.dirname(f))
logging.info('found %d spm.mat files' % len(path_spm_mat))
logging.info('paths to spm.mat files:\n%s' % pformat(path_spm_mat))
dl.get(path_spm_mat)

# load the t-maps of the first-level glm:
path_tmap = opj(path_level1, 'contrasts', sub, '*', 'spmT*.nii')
path_tmap = sorted(glob.glob(path_tmap), key=lambda f: os.path.dirname(f))
logging.info('found %d t-maps' % len(path_tmap))
logging.info('paths to t-maps files:\n%s' % pformat(path_tmap))
dl.get(path_tmap)
'''
========================================================================
LOAD THE MRI DATA:
========================================================================
'''
anat = image.load_img(path_anat[0])
logging.info('successfully loaded %s' % path_anat[0])
# load visual mask:
mask_vis = image.load_img(path_mask_vis_task[0]).get_data().astype(int)
logging.info('successfully loaded one visual mask file!')
# load tmap data:
tmaps = [image.load_img(i).get_data().astype(float) for i in copy.deepcopy(path_tmap)]
logging.info('successfully loaded the tmaps!')
# load hippocampus mask:
mask_hpc = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_hpc_task)]
logging.info('successfully loaded one visual mask file!')
'''
FEATURE SELECTION
'''
# plot raw-tmaps on an anatomical background:
logging.info('plotting raw tmaps with anatomical as background:')
for i, path in enumerate(path_tmap):
    logging.info('plotting raw tmap %s (%d of %d)' % (path, i+1, len(path_tmap)))
    path_save = opj(path_out_figs, '%s_run-%02d_tmap_raw.png' % (sub, i+1))
    plotting.plot_roi(path, anat, title=os.path.basename(path_save),
                      output_file=path_save, colorbar=True)
