

#build classification model
import copy
from utils.encoder_utils import get_preprocessor
import tensorflow as tf, tensorflow_hub as hub

def BertClassifier(num_classes, dropout,
                   task_target_sizes, main_task_names,
                   pooled_task_name='000',
                   ablate=[]):

    class Classifier(tf.keras.Model):

        def __init__(self, num_classes):
            super(Classifier, self).__init__(name = "prediction")
            share_sizes = [256, 128]
            layer_sizes = [128, 64]
            sublayer_sizes = [64, 32]

            # shared generators
            if 'shared' not in ablate:
                shared_dense = [tf.keras.layers.Dense(share_sizes[l_idx//2], activation='relu', name=f'shared_dense_{l_idx}')
                            for l_idx in range(len(share_sizes)*2)]
                shared_dropout = [tf.keras.layers.Dropout(dropout, name=f'shared_dropout_{l_idx}')
                                for l_idx in range(len(share_sizes)*2)]
                shared_layers = [sub for sub in zip(shared_dense, shared_dropout)] # we alternate dense and dropout layers
            else:
                shared_layers = []


            # task generators
            if 'task-type' not in ablate:
                tasks_dense = [tf.keras.layers.Dense(layer_sizes[l_idx//2], activation='relu', name=f'task_dense_{l_idx}')
                            for l_idx in range(len(layer_sizes)*2)]
                tasks_dropout = [tf.keras.layers.Dropout(dropout, name=f'task_dropout_{l_idx}')
                                for l_idx in range(len(layer_sizes)*2)]
                tasks_layers_generated = [sub for sub in zip(tasks_dense, tasks_dropout)] # we alternate dense and dropout layers
            else:
                tasks_layers_generated = []

            # subtask generators
            if 'task' not in ablate:
                subtasks_dense = [tf.keras.layers.Dense(sublayer_sizes[l_idx//2], activation='relu', name=f'subtask_dense_{l_idx}')
                                for l_idx in range(len(sublayer_sizes)*2)]
                subtasks_dropout = [tf.keras.layers.Dropout(dropout, name=f'subtask_dropout_{l_idx}')
                                    for l_idx in range(len(sublayer_sizes)*2)]
                subtasks_layers_generated = [sub for sub in zip(subtasks_dense, subtasks_dropout)] # we alternate dense and dropout layers
            else:
                subtasks_layers_generated = []

            # ------ Shared layers
            self.shared_layers = shared_layers
            self.task_layers = []
            self.subtask_layers = []
            self.subtask_acts = []

            # ------ task layers
            for task_target_size in task_target_sizes:
                self.task_layers += [copy.deepcopy(tasks_layers_generated)]
                self.subtask_layers += [[copy.deepcopy(subtasks_layers_generated) for _ in range(task_target_size)]]
                self.subtask_acts +=  [[tf.keras.layers.Dense(1, activation='sigmoid', name=f'subtask_act_{sub_idx}')
                                        for sub_idx in range(task_target_size)]]


        def call(self, preprocessed_text):
            x = tf.identity(preprocessed_text)

            # ======= shared =======
            for layer, _ in self.shared_layers:
                x = layer(x)
            shared_rep = tf.identity(x)


            # ======= tasks =======
            reps_maintasks = []
            outputs_ti = []

            # ---- main tasks (first branching)
            for taski_layers in self.task_layers:
                x = tf.identity(shared_rep)
                for layer, _ in taski_layers:
                    x = layer(x)
                reps_maintasks += [tf.identity(x)]

            # ---- subtasks (second branching)
            for t_idx, rep_ti in enumerate(reps_maintasks):
                output_subti = []
                for sub_idx, sub_layers in enumerate(self.subtask_layers[t_idx]):
                    x = tf.identity(rep_ti)
                    for layer, _ in sub_layers:
                        x = layer(x)
                    output_subti += [self.subtask_acts[t_idx][sub_idx](x)]

                if (main_task_names[t_idx]==pooled_task_name):
                    outputs_ti += [tf.expand_dims(tf.reduce_max(tf.keras.layers.Concatenate(axis=1)(output_subti),axis=1), axis=-1)]
                else:
                    outputs_ti += [tf.keras.layers.Concatenate(axis=1)(output_subti)]

            output = tf.keras.layers.Concatenate(axis=1)(outputs_ti)

            return output



    return Classifier(num_classes)

"""### Auxillary model
This model is used only for testing.
It uses the pretrained layers from the main model
to give the predictions, plus intermediate layer outputs
"""

#build classification model
import copy

def aux_BertClassifier(layers_list, layer_names,
                       task_sizes, main_task_names, pooled_task_name='000',
                       stage_name='shared'):

    class Classifier(tf.keras.Model):

        def __init__(self):
            num_classes=2
            super(Classifier, self).__init__(name = "prediction")

            shared_idx = max([lid for lid, l in enumerate(layer_names) if l[:6]=='shared']) + 1
            task_idx = max([lid for lid, l in enumerate(layer_names) if l[:4]=='task']) + 1
            subtask_idx = min([lid for lid, l in enumerate(layer_names) if l[:11]=='subtask_act'])

            lcount = (task_idx-shared_idx) // len(task_sizes)

            self.encoder = layers_list[0]
            self.shared_layers = layers_list[1:shared_idx]
            self.task_layers = [layers_list[shared_idx+t_idx*lcount:shared_idx+(t_idx+1)*lcount]
                                for t_idx in range(len(task_sizes))]
            self.subtask_layers = []
            self.subtask_acts = []
            idx0 = task_idx
            idx1 = subtask_idx
            for t_idx, t_size in enumerate(task_sizes):
                idx0 += t_idx * lcount
                self.subtask_layers += [[layers_list[idx0+lcount*subt_idx:idx0+lcount*(subt_idx+1)]
                                    for subt_idx in range(t_size)]]
                self.subtask_acts +=  [layers_list[idx1:idx1+t_size]]
                idx1 += t_size



        def call(self, preprocessed_text):
            encoder_output = self.encoder(preprocessed_text)

            # ======= shared =======
            x = tf.identity(encoder_output["pooled_output"])
            for layer in self.shared_layers:
                x = layer(x)
            shared_rep = tf.identity(x)


            # ======= tasks =======
            reps_maintasks = []
            reps_subtasks = []
            outputs_ti = []

            # ---- main tasks (first branching)
            for taski_layers in self.task_layers:
                x = tf.identity(shared_rep)
                for layer in taski_layers:
                    x = layer(x)
                reps_maintasks += [tf.identity(x)]

            # ---- subtasks (second branching)
            for t_idx, rep_ti in enumerate(reps_maintasks):
                output_subti = []
                for sub_idx, sub_layers in enumerate(self.subtask_layers[t_idx]):
                    x = tf.identity(rep_ti)
                    for layer in sub_layers:
                        x = layer(x)
                    reps_subtasks += [tf.identity(x)]
                    output_subti += [self.subtask_acts[t_idx][sub_idx](x)]

                if (main_task_names[t_idx]==pooled_task_name):
                    outputs_ti += [tf.expand_dims(tf.reduce_max(tf.keras.layers.Concatenate(axis=1)(output_subti),axis=1), axis=-1)]
                else:
                    outputs_ti += [tf.keras.layers.Concatenate(axis=1)(output_subti)]

            output = tf.keras.layers.Concatenate(axis=1)(outputs_ti)

            return {'final_output': output,
                    'BERT_output': encoder_output["pooled_output"],
                    'shared_rep': shared_rep,
                    'reps_maintasks': reps_maintasks,
                    'reps_subtasks': reps_subtasks}

    return Classifier()

"""# Training

## Training parameters
"""