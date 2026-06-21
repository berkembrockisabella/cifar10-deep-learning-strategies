import os
import gc
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mob
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_res

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.environ.setdefault('DML_VISIBLE_DEVICES', '0')

CLASS_NAMES = ['aviao', 'automovel', 'passaro', 'gato', 'cervo',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao']

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, 'resultados.csv')


# =============================================================================
# Registro de resultados em CSV
# =============================================================================
def inicializar_csv():
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'estrategia', 'variante', 'acuracia',
            'precision_macro', 'recall_macro', 'f1_macro',
            'timestamp'
        ])


def registrar_resultado(estrategia, variante, y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            estrategia, variante,
            f'{acc*100:.2f}', f'{p*100:.2f}', f'{r*100:.2f}', f'{f*100:.2f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])
    return acc


# =============================================================================
# Funções auxiliares
# =============================================================================
def plot_cm(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(titulo)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    nome_arquivo = titulo.replace(' ', '_').replace('|', '').replace('/', '-')[:60]
    plt.savefig(os.path.join(FIGURES_DIR, f'{nome_arquivo}.png'), dpi=100)
    plt.close()


def resize_uint8(X, size):
    return np.array([cv2.resize(img, (size, size)) for img in X], dtype='uint8')


def callbacks_padrao():
    return [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=0)
    ]


def liberar_memoria(*objetos):
    for obj in objetos:
        del obj
    gc.collect()
    tf.keras.backend.clear_session()


# =============================================================================
# Carregamento do CIFAR-10
# =============================================================================
print('=' * 60)
print('Carregando CIFAR-10...')
print('=' * 60)

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()

y_train = y_train_raw.ravel()
y_test  = y_test_raw.ravel()

y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

print(f'Treino : {X_train_raw.shape}')
print(f'Teste  : {X_test_raw.shape}')

inicializar_csv()


# =============================================================================
# Estratégia 1 — CNN treinada do zero
# =============================================================================
print('\n' + '=' * 60)
print('Estratégia 1 — CNN treinada do zero')
print('=' * 60)

X_train_s1 = X_train_raw.astype('float32') / 255.0
X_test_s1  = X_test_raw.astype('float32')  / 255.0

model_s1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
], name='CNN_do_zero')

model_s1.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

model_s1.fit(
    X_train_s1, y_train_cat,
    epochs=50, batch_size=64,
    validation_split=0.1,
    callbacks=callbacks_padrao(),
    verbose=1
)

y_pred_s1 = np.argmax(model_s1.predict(X_test_s1), axis=1)
acc_s1 = registrar_resultado('Estratégia 1', 'CNN do zero', y_test, y_pred_s1)
print(f'\nEstratégia 1 — Acurácia: {acc_s1*100:.2f}%')
print(classification_report(y_test, y_pred_s1, target_names=CLASS_NAMES))
plot_cm(y_test, y_pred_s1, f'Estratégia 1 — CNN do zero | {acc_s1*100:.2f}%')

del X_train_s1, X_test_s1
liberar_memoria(model_s1)


# =============================================================================
# Estratégia 2 — Extração de características + classificador raso
# Questão 2a: MobileNetV2 (leve) vs ResNet50 (pesada)
# =============================================================================
print('\n' + '=' * 60)
print('Estratégia 2 — Extração de características')
print('=' * 60)

X_train_96 = resize_uint8(X_train_raw, 96)
X_test_96  = resize_uint8(X_test_raw,  96)
print(f'Shape uint8 treino: {X_train_96.shape} | RAM: {X_train_96.nbytes / 1e6:.0f} MB')


def extrair_features(model_fn, preprocess_fn, X_tr96, X_te96, nome):
    X_tr = preprocess_fn(X_tr96.astype('float32'))
    X_te = preprocess_fn(X_te96.astype('float32'))

    extractor = model_fn(
        include_top=False, weights='imagenet', pooling='avg',
        input_tensor=Input(shape=(96, 96, 3))
    )
    extractor.trainable = False

    F_train = extractor.predict(X_tr, batch_size=64, verbose=1)
    F_test  = extractor.predict(X_te, batch_size=64, verbose=1)

    del extractor, X_tr, X_te
    gc.collect()
    tf.keras.backend.clear_session()

    return F_train, F_test


def classificar_features(F_train, F_test, y_train, y_test, nome):
    params = {
        'hidden_layer_sizes': [(256,), (512,), (256, 128)],
        'activation': ['relu', 'logistic']
    }
    gs = GridSearchCV(
        MLPClassifier(max_iter=500, random_state=42),
        params, cv=3, n_jobs=-1, verbose=1
    )
    gs.fit(F_train, y_train)
    print(f'{nome} — melhores parâmetros: {gs.best_params_}')

    y_pred = gs.best_estimator_.predict(F_test)
    acc = registrar_resultado('Estratégia 2', nome, y_test, y_pred)
    print(f'{nome} — acurácia no teste: {acc*100:.2f}%')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    plot_cm(y_test, y_pred, f'Estratégia 2 — {nome} | {acc*100:.2f}%')
    return acc


F_train_mob, F_test_mob = extrair_features(MobileNetV2, preprocess_mob, X_train_96, X_test_96, 'MobileNetV2')
F_train_res, F_test_res = extrair_features(ResNet50,    preprocess_res, X_train_96, X_test_96, 'ResNet50')

acc_mob = classificar_features(F_train_mob, F_test_mob, y_train, y_test, 'MobileNetV2')
acc_res = classificar_features(F_train_res, F_test_res, y_train, y_test, 'ResNet50')

print(f'\nMobileNetV2 : {acc_mob*100:.2f}%')
print(f'ResNet50    : {acc_res*100:.2f}%')
print(f'Diferença   : {abs(acc_res - acc_mob)*100:.2f} p.p.')

del F_train_mob, F_test_mob, F_train_res, F_test_res
gc.collect()


# =============================================================================
# Estratégia 3 — Fine-tuning da ResNet50
# Fase 1: base congelada | Fase 2: últimas camadas descongeladas
# =============================================================================
print('\n' + '=' * 60)
print('Estratégia 3 — Fine-tuning ResNet50')
print('=' * 60)

X_train_ft = preprocess_res(X_train_96.astype('float32'))
X_test_ft  = preprocess_res(X_test_96.astype('float32'))

base_s3 = ResNet50(weights='imagenet', include_top=False,
                   input_tensor=Input(shape=(96, 96, 3)))
base_s3.trainable = False

x   = base_s3.output
x   = GlobalAveragePooling2D()(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.5)(x)
out = Dense(10, activation='softmax')(x)

model_s3 = Model(inputs=base_s3.input, outputs=out, name='FineTuning_ResNet50')
model_s3.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

print('Fase 1: treinando apenas o topo')
h_s3_f1 = model_s3.fit(
    X_train_ft, y_train_cat,
    epochs=20, batch_size=64,
    validation_split=0.1,
    callbacks=callbacks_padrao(),
    verbose=1
)

base_s3.trainable = True
for layer in base_s3.layers[:140]:
    layer.trainable = False

model_s3.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print('Fase 2: fine-tuning das camadas superiores')
h_s3_f2 = model_s3.fit(
    X_train_ft, y_train_cat,
    epochs=20, batch_size=32,
    validation_split=0.1,
    callbacks=callbacks_padrao(),
    verbose=1
)

y_pred_s3 = np.argmax(model_s3.predict(X_test_ft), axis=1)
acc_s3 = registrar_resultado('Estratégia 3', 'Fine-Tuning ResNet50', y_test, y_pred_s3)
print(f'\nEstratégia 3 — Acurácia: {acc_s3*100:.2f}%')
print(classification_report(y_test, y_pred_s3, target_names=CLASS_NAMES))
plot_cm(y_test, y_pred_s3, f'Estratégia 3 — Fine-Tuning ResNet50 | {acc_s3*100:.2f}%')

val_acc = h_s3_f1.history['val_accuracy'] + h_s3_f2.history['val_accuracy']
sep     = len(h_s3_f1.history['val_accuracy'])
plt.figure(figsize=(9, 4))
plt.plot(val_acc, label='Val Accuracy')
plt.axvline(sep, color='red', linestyle='--', label='Início fine-tuning')
plt.title('Estratégia 3 — Acurácia de Validação')
plt.xlabel('Época')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'S3_curva_validacao.png'), dpi=100)
plt.close()

del X_train_ft, X_test_ft
liberar_memoria(model_s3)


# =============================================================================
# Estratégia 4 — Fine-tuning com data augmentation
# Questão 4a: Flatten vs GlobalMaxPooling2D
# Questão 4b: Adam vs SGD
# Questão 4c: augmentation básica vs agressiva
# =============================================================================
print('\n' + '=' * 60)
print('Estratégia 4 — Data augmentation')
print('=' * 60)

X_train_da = preprocess_mob(X_train_96.astype('float32'))
X_test_da  = preprocess_mob(X_test_96.astype('float32'))

aug_basica = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, zoom_range=0.1
)

aug_agressiva = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    horizontal_flip=True, zoom_range=0.2, shear_range=0.1,
    brightness_range=[0.8, 1.2]
)


def experimento_s4(pooling_tipo, optimizer_fn, aug_gen, nome):
    print(f'\n--- {nome} ---')

    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_tensor=Input(shape=(96, 96, 3)))
    base.trainable = False

    x = base.output
    if pooling_tipo == 'flatten':
        x = Flatten()(x)
    elif pooling_tipo == 'gmp':
        x = GlobalMaxPooling2D()(x)
    else:
        x = GlobalAveragePooling2D()(x)

    x   = Dense(256, activation='relu')(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.5)(x)
    out = Dense(10, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        aug_gen.flow(X_train_da, y_train_cat, batch_size=64),
        epochs=15, validation_data=(X_test_da, y_test_cat),
        callbacks=callbacks_padrao(), verbose=0
    )

    base.trainable = True
    for layer in base.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=optimizer_fn(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        aug_gen.flow(X_train_da, y_train_cat, batch_size=32),
        epochs=20, validation_data=(X_test_da, y_test_cat),
        callbacks=callbacks_padrao(), verbose=0
    )

    y_pred = np.argmax(model.predict(X_test_da), axis=1)
    acc = registrar_resultado('Estratégia 4', nome, y_test, y_pred)
    print(f'Acurácia: {acc*100:.2f}%')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    plot_cm(y_test, y_pred, f'Estratégia 4 — {nome} | {acc*100:.2f}%')

    del model, base
    gc.collect()
    tf.keras.backend.clear_session()

    return acc


resultados_s4 = {}
resultados_s4['Flatten + Adam + AugBasica'] = experimento_s4(
    'flatten', lambda: Adam(1e-5), aug_basica, 'Flatten + Adam + AugBasica')
resultados_s4['GMP + Adam + AugBasica'] = experimento_s4(
    'gmp', lambda: Adam(1e-5), aug_basica, 'GlobalMaxPooling2D + Adam + AugBasica')
resultados_s4['GAP + SGD + AugBasica'] = experimento_s4(
    'gap', lambda: SGD(1e-4, momentum=0.9), aug_basica, 'GAP + SGD + AugBasica')
resultados_s4['GAP + Adam + AugAgressiva'] = experimento_s4(
    'gap', lambda: Adam(1e-5), aug_agressiva, 'GAP + Adam + AugAgressiva')

print('\nResumo Estratégia 4:')
for nome, acc in resultados_s4.items():
    print(f'  {nome}: {acc*100:.2f}%')

del X_train_da, X_test_da
gc.collect()


# =============================================================================
# Estratégia 5 — Fine-tuning EfficientNetV2B0 (bônus)
# =============================================================================
print('\n' + '=' * 60)
print('Estratégia 5 — Fine-tuning EfficientNetV2B0 (bônus)')
print('=' * 60)

from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_eff

X_train_vit = preprocess_eff(X_train_96.astype('float32'))
X_test_vit  = preprocess_eff(X_test_96.astype('float32'))

base_vit = EfficientNetV2B0(
    weights='imagenet', include_top=False, pooling='avg',
    input_tensor=Input(shape=(96, 96, 3))
)
base_vit.trainable = False

x   = base_vit.output
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(10, activation='softmax')(x)

model_vit = Model(inputs=base_vit.input, outputs=out, name='EfficientNetV2B0')
model_vit.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

print('Fase 1: treinando apenas o topo')
model_vit.fit(
    X_train_vit, y_train_cat,
    epochs=10, batch_size=64,
    validation_split=0.1,
    callbacks=callbacks_padrao(),
    verbose=1
)

base_vit.trainable = True
freeze_until = int(len(base_vit.layers) * 0.8)
for layer in base_vit.layers[:freeze_until]:
    layer.trainable = False

model_vit.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print('Fase 2: fine-tuning das camadas superiores')
model_vit.fit(
    X_train_vit, y_train_cat,
    epochs=15, batch_size=64,
    validation_split=0.1,
    callbacks=callbacks_padrao(),
    verbose=1
)

y_pred_vit = np.argmax(model_vit.predict(X_test_vit), axis=1)
acc_vit = registrar_resultado('Estratégia 5', 'EfficientNetV2B0', y_test, y_pred_vit)
print(f'\nEstratégia 5 — Acurácia: {acc_vit*100:.2f}%')
print(classification_report(y_test, y_pred_vit, target_names=CLASS_NAMES))
plot_cm(y_test, y_pred_vit, f'Estratégia 5 — EfficientNetV2B0 | {acc_vit*100:.2f}%')

del X_train_vit, X_test_vit
liberar_memoria(model_vit)


# =============================================================================
# Resumo geral
# =============================================================================
print('\n' + '=' * 60)
print('RESUMO GERAL — CIFAR-10')
print('=' * 60)
print(f'Estratégia 1  — CNN do zero              : {acc_s1*100:.2f}%')
print(f'Estratégia 2  — Feature Ext. MobileNetV2 : {acc_mob*100:.2f}%')
print(f'Estratégia 2  — Feature Ext. ResNet50    : {acc_res*100:.2f}%')
print(f'Estratégia 3  — Fine-Tuning ResNet50     : {acc_s3*100:.2f}%')
for nome, acc in resultados_s4.items():
    print(f'Estratégia 4  — {nome}: {acc*100:.2f}%')
print(f'Estratégia 5  — EfficientNetV2B0 (bônus) : {acc_vit*100:.2f}%')
print('=' * 60)
print(f'\nResultados salvos em: {CSV_PATH}')
