TARGET_GROUPS = [
'Performance',
'jpinpoint-common-rules',
'Best Practices',
'Multithreading',
'jpinpoint-concurrent-rules',
]

TOOLS = ['BugLocator',
         'BRTracer',
         'BLIA']


PREDICTED_SMELLS = 'grouped'
BUG_SMELL_FEATURES = 'biggest_file_prod_code_prio'

EMBEDDING = 'stackoverflow_mpnet_embeddings'
DOC = 'full'
DROPOUT = 0.1
LOSS_FUNCTION = 'weighted'
HIDDEN_LAYERS = 1
LAYER_WIDTH = 300
BATCH_SIZE = 32
EPOCHS = 100


BOOTSTRAP_REPEATS = 20
BOOTSTRAP_SAMPLE_FRACTION = 0.8