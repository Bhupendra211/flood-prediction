import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping




def train_xgboost(X, y, save_path=None):
# simple time-based split: keep last 20% as test
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]


    model = XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)


    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, preds))
    print('ROC AUC:', roc_auc_score(y_test, proba))


    if save_path:
        joblib.dump(model, save_path)
    return model




def build_lstm_model(input_shape, units=64):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




def train_lstm(X_seq, y_seq, save_path=None, epochs=30, batch_size=32):
# time-split
    split = int(len(X_seq)*0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]


    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[es])


    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'LSTM test loss: {loss:.4f}, acc: {acc:.4f}')


    if save_path:
        model.save(save_path)
    return model