import tensorflow as tf
from train_test.loss import get_loss
from utils.batch_embeddings_processors import *
from official.nlp import optimization
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report



def train(strategy, bert_classifier,
          ff_train, ff_val,
          task_sizes, main_task_names, num_tasks,
          txt2embed, embedding_dim=4096,
          epochs=10, init_lr=3e-4, wt_decay_rate=.01, warmup_pct=5, dropout=.4, batch_size=256,
          workers=1, seq_length=tf.constant(512)):

    #define initial best parameters and maximum accuracy
    max_wf1 = 0
    pars_best = None

    with strategy.scope():
        # specify metrics and loss
        # take the max over each subset of columns (for each main task)
        loss = get_loss(strategy, ff_train, task_sizes, main_task_names)

        metrics = [tf.keras.metrics.BinaryAccuracy(name = "accuracy"),
                tf.keras.metrics.Precision(name = "precision"),
                tf.keras.metrics.Recall(name = "recall")]

        print(f"Training init_lr = {init_lr}, wt_decay_rate = {wt_decay_rate}, warmup_pct = {warmup_pct}, dropout = {dropout}, batch_size = {batch_size}...")

        #preprocess data
        data_train, data_train_n = load_batch_process(ff_train, batch_size,
                                                    txt2embed,
                                                    task_sizes, main_task_names, num_tasks,
                                                    embedding_dim=embedding_dim)
        data_val, data_val_n = load_batch_process(ff_val, batch_size, txt2embed,
                                                task_sizes, main_task_names, num_tasks,
                                                embedding_dim=embedding_dim)
        #data_test, y_test = load_batch_test(ff_test, batch_size, bert_preprocess_model)

        #calculate appropriate settings for optimization and fitting
        steps_per_epoch = data_train_n // batch_size
        validation_steps = data_val_n // batch_size
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = num_train_steps // warmup_pct

        #instantiate linear decay and warmup schedules
        decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate = init_lr,
                                                                        decay_steps = num_train_steps,
                                                                        end_learning_rate = 0)

        warmup_schedule = optimization.WarmUp(initial_learning_rate = init_lr,
                                            decay_schedule_fn = decay_schedule,
                                            warmup_steps = num_warmup_steps)

        #instantiate AdamW optimizer with scheduled warmup and decay
        optimizer = optimization.AdamWeightDecay(learning_rate = warmup_schedule,
                                                weight_decay_rate = wt_decay_rate)

        #compile classification model
        bert_classifier.compile(optimizer, loss, metrics)


        #configure early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
                                                min_delta = 0.001,
                                                mode = "min",
                                                restore_best_weights = True)

        #fit classification model
        history = bert_classifier.fit(data_train.repeat(),
                                    batch_size = batch_size,
                                    epochs = epochs, validation_data = data_val.repeat(),
                                    steps_per_epoch = steps_per_epoch,
                                    validation_steps = validation_steps,
                                    validation_batch_size = batch_size, workers = workers,
                                    callbacks = [early_stop])

        data_val2, y_val = load_batch_test(ff_val, batch_size, txt2embed,
                                            task_sizes, main_task_names, num_tasks,
                                            embedding_dim=embedding_dim)

        y_val_pred = bert_classifier.predict(data_val2, batch_size = batch_size, workers = workers)

        #find best threshold and calculate AUC
        combined_acc = []
        combined_thresh = []

        for i in range(y_val.shape[1]):
            y_val_i = y_val[:, i]
            y_val_pred_i = y_val_pred[:, i]
            ids_task = np.where(~(y_val_i==-1))
            y_val_task, y_val_pred_task = y_val_i[ids_task], y_val_pred_i[ids_task]

            fpr, tpr, thresh = roc_curve(y_val_task, y_val_pred_task)
            J = tpr - fpr
            J_max = np.argmax(J)
            thresh_best = thresh[J_max]
            combined_thresh.append(thresh_best)

            y_val_pred_task = y_val_pred_task > thresh_best
            cv_wf1 = f1_score(y_val_task, y_val_pred_task, average='weighted')
            combined_acc.append(cv_wf1)

        cv_wf1 = sum(combined_acc) / len(combined_acc) #average accuracy across all tasks - maybe not the best for hyperparameter tuning?
        #update maximum accuracy and best parameters
        if cv_wf1 > max_wf1:
            max_wf1 = cv_wf1
            pars_best = [init_lr, wt_decay_rate, warmup_pct, dropout, batch_size, combined_thresh]

        print("Val weighted f1: {}".format(max_wf1))


    return bert_classifier, pars_best




def test(strategy, bert_classifier,
         ff_test, pars_best,
         task_list, task_sizes, main_task_names, num_tasks,
         txt2embed, embedding_dim=4096,
         pooled_task_name="000",
         batch_size=256, 
         workers=1):
    
    metrics_dict = dict()
    metrics_str = ""

    if main_task_names[0]==pooled_task_name:
        task_names = [pooled_task_name] + task_list[task_sizes[0]:]
    else:
        task_names = task_list

    with strategy.scope():

        data_test, y_test = load_batch_test(ff_test, batch_size, txt2embed,
                                            task_sizes, main_task_names, num_tasks,
                                            embedding_dim=embedding_dim)

        #produce test set predictions
        y_test_pred = bert_classifier.predict(data_test, batch_size = pars_best[4], workers = workers)

        #calculate AUC

        combined_thresh = pars_best[5]
        #combined_acc = []

        # Calculate metrics
        for i in range(y_test.shape[1]):
            #convert to class labels using best threshold and calculate performance metrics
            ids_task = np.where(~(y_test[:,i]==-1))
            y_test_task, y_test_pred_task = y_test[ids_task][:,i], y_test_pred[ids_task][:,i]
            y_test_pred_task = y_test_pred_task > combined_thresh[i]

            # fpr_DET, fnr, _ = roc_curve(y_test_task, y_test_pred_task)
            # prec, rec, _ = precision_recall_curve(y_test_task, y_test_pred_task)
            # fpr, tpr, _ = roc_curve(y_test_task, y_test_pred_task)


            # BERT_ROC = pd.DataFrame({f"BERT_fpr_{i}": fpr, f"BERT_tpr_{i}": tpr}) #once hyperparameter tuning is complete, these lines can be used to output ROC, precision/recall, etc. curves by simply sending it to a plotting function or printing to CSV
            # BERT_DET = pd.DataFrame({f"BERT_fpr_{i}": fpr_DET, f"BERT_fnr_{i}": fnr})
            # BERT_PR = pd.DataFrame({f"BERT_prec_{i}": prec, f"BERT_rec_{i}": rec})

            print(task_names[i])

            try:
                auc = roc_auc_score(y_test_task, y_test_pred_task)
                print("AUC: {}".format(auc))
            except:
                print("AUC: undefined")

            report_str = classification_report(y_test_task, y_test_pred_task, digits = 3)
            print(report_str)
            metrics_str += report_str + "\n"

            metrics_dict[task_names[i]] = classification_report(y_test_task, y_test_pred_task,
                                                                digits = 3, output_dict=True)
            
    
    return metrics_dict, metrics_str
