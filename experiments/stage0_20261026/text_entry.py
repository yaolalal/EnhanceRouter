from common import *
from frontend import text
text(DS[int(os.environ['SLURM_ARRAY_TASK_ID'])],extra='--extra' in sys.argv)
