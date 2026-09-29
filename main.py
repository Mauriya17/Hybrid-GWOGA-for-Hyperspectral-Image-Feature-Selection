# ===========================================================================
# 1. SETUP & IMPORTS
# ===========================================================================
import os
import pickle
import warnings
import cv2
import numpy as np
import pandas as pd
import time
import zipfile 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import sys 

# --- Suppress All Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR') 

from tensorflow.keras.applications import ConvNeXtBase 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical 

# ===========================================================================
# 2. CONFIGURATION
# ===========================================================================
class Config:
    """Holds all configuration parameters for the project."""
    
    BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    
    DATASET_CATALOG = {
        "Medical_Waste_4.0": {
            "zip_name": "Medical Waste 4.0.zip",
            "image_path_relative": os.path.join('Medical Waste 4.0', 'Medical Waste 4.0'), 
            "FINETUNE_EPOCHS": 8, "GWO_MAX_ITER": 25, "GA_GENERATIONS": 25
        },
        "XRay_Fracture": {
            "zip_name": "X-ray Bone Fracture Dataset Comprehensive Imaging for Fracture Classification and Medical Research.zip",
            "image_path_relative": os.path.join('X-ray Bone Fracture Dataset Comprehensive Imaging for Fracture Classification and Medical Research', 'Bone -Fracture.zip', 'Bone Fracture', 'Augmented'),
            "FINETUNE_EPOCHS": 8, "GWO_MAX_ITER": 25, "GA_GENERATIONS": 25
        },
        "UC_Merced": {
            "zip_name": "archive.zip", 
            "image_path_relative": os.path.join('UCMerced_LandUse', 'Images'),
            "FINETUNE_EPOCHS": 8, "GWO_MAX_ITER": 25, "GA_GENERATIONS": 25
        }
    }
    
    IMAGE_SIZE = (224, 224)
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    FINETUNE_BATCH_SIZE = 32
    L2_REG_FACTOR = 0.001
    EARLY_STOPPING_PATIENCE = 5 
    FITNESS_ERROR_WEIGHT = 0.99
    GWO_N_WOLVES = 8
    GA_POPULATION_SIZE = 16
    GA_MUTATION_RATE = 0.1
    SVM_KERNEL = 'rbf'
    SVM_C = 1.0
    OUTPUT_DIR = 'project_output'
    MIXUP_ALPHA = 0.4 


# ===========================================================================
# 3. STYLED OUTPUT FUNCTIONS
# ===========================================================================
class Style:
    """ANSI escape codes for terminal colors."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    RED = '\033[31m'

def print_header(text, color=Style.CYAN):
    """
    Prints a styled header to the console.
    
    Args:
        text (str): The text to display in the header.
        color (str, optional): ANSI color code from the Style class. 
                              Defaults to Style.CYAN.
    """
    print("\n" + color + Style.BOLD + "="*80 + Style.RESET)
    print(color + Style.BOLD + f" {text} ".center(80, '=') + Style.RESET)
    print(color + Style.BOLD + "="*80 + Style.RESET)

def print_info(text):
    """
    Prints an informational message to the console.
    
    Args:
        text (str): The message to display.
    """
    print(f"   {Style.YELLOW}[i]{Style.RESET} {text}")

def print_success(text):
    """
    Prints a success message to the console.
    
    Args:
        text (str): The message to display.
    """
    print(f"   {Style.GREEN}[✓]{Style.RESET} {text}")

def print_error(text):
    """
    Prints an error message to the console.
    
    Args:
        text (str): The message to display.
    """
    print(f"   {Style.RED}[ERROR]{Style.RESET} {text}")
    
# ===========================================================================
# 4. UTILITY FUNCTIONS
# ===========================================================================

def auto_unzip_and_get_path(base_dir, zip_name, expected_image_path):
    """
    Resolves dataset paths and performs single or nested ZIP extraction.

    This function checks if the target dataset path exists. If not, it
    attempts to extract an outer ZIP file. If the `expected_image_path`
    contains a nested `.zip` file, it will also extract the inner ZIP.

    Args:
        base_dir (str): The root directory where datasets are stored.
        zip_name (str): The filename of the primary (outer) ZIP file.
        expected_image_path (str): The relative path *within* the base_dir
                                   (post-extraction) where the images are
                                   expected, which may contain a nested ZIP.

    Returns:
        str: The absolute path to the final image directory if successful.
        None: If any path or extraction fails.
    """
    
    path_parts_no_zip = [
        part
        for part in expected_image_path.replace("\\", "/").split("/")
        if not part.lower().endswith(".zip")
    ]
    correct_final_path = os.path.join(base_dir, *path_parts_no_zip)

    if os.path.exists(correct_final_path) and os.path.isdir(correct_final_path):
        print_success(f"Dataset path found: {correct_final_path}")
        return correct_final_path

    print_info(f"Processing outer ZIP: {zip_name}")
    outer_zip_file_path = os.path.join(base_dir, zip_name)
    if not os.path.exists(outer_zip_file_path):
        print_error(f"Outer ZIP not found: {outer_zip_file_path}")
        return None

    try:
        print_info(f"Extracting outer ZIP: {outer_zip_file_path}...")
        with zipfile.ZipFile(outer_zip_file_path, "r") as zr:
            zr.extractall(base_dir) 
        print_success("Outer ZIP extraction complete.")
    except Exception as e:
        print_error(f"Outer ZIP extraction failed: {e}")
        return None

    path_parts = expected_image_path.replace("\\", "/").split("/")
    inner_zip_index = next(
        (i for i, part in enumerate(path_parts) if part.lower().endswith(".zip")), -1
    )
    
    if inner_zip_index != -1:
        inner_zip_filename = path_parts[inner_zip_index]
        inner_zip_file_path = os.path.join(base_dir, *path_parts[: inner_zip_index + 1])
        inner_extract_target_dir = os.path.join(base_dir, *path_parts[:inner_zip_index])
        
        print_info(f"Nested ZIP detected: {inner_zip_filename}")
        
        if not os.path.exists(inner_zip_file_path):
            print_error(f"Inner ZIP file NOT found after outer extraction: {inner_zip_file_path}")
            return None
        
        try:
            print_info(f"Extracting inner ZIP to: {inner_extract_target_dir}...")
            with zipfile.ZipFile(inner_zip_file_path, "r") as zr:
                zr.extractall(inner_extract_target_dir)
            print_success("Inner ZIP extraction complete.")
        except Exception as e:
            print_error(f"Inner ZIP extraction failed: {e}")
            return None

    if os.path.exists(correct_final_path) and os.path.isdir(correct_final_path):
        print_success(f"Verified final dataset path: {correct_final_path}")
        return correct_final_path
    else:
        print_error(f"Verification failed. Path NOT found: {correct_final_path}")
        return None


def mixup_data(X_batch, y_batch_int, num_classes_total, alpha=Config.MIXUP_ALPHA):
    """
    Applies Mixup augmentation to a batch of images and labels.

    Mixup creates new training samples by linearly interpolating two
    random samples from the batch.

    Args:
        X_batch (np.ndarray): The batch of images.
        y_batch_int (np.ndarray): The batch of integer-encoded labels.
        num_classes_total (int): The total number of classes in the dataset.
        alpha (float, optional): The alpha parameter for the Beta distribution
                                 used to generate the interpolation factor (lambda).
                                 Defaults to Config.MIXUP_ALPHA.

    Returns:
        tuple: A tuple containing:
            - (np.ndarray): The mixed image batch.
            - (np.ndarray): The mixed, one-hot encoded (soft) label batch.
    """
    if len(X_batch) < 2:
        return X_batch, to_categorical(y_batch_int, num_classes=num_classes_total)
        
    lam = np.random.beta(alpha, alpha)
    
    indices = np.arange(len(X_batch))
    np.random.shuffle(indices)
    
    X_mixed = lam * X_batch + (1 - lam) * X_batch[indices]
    
    y_one_hot = to_categorical(y_batch_int, num_classes=num_classes_total)
    y_shuffled_one_hot = to_categorical(y_batch_int[indices], num_classes=num_classes_total)
    
    y_mixed = lam * y_one_hot + (1 - lam) * y_shuffled_one_hot
    
    return X_mixed.astype(np.float32), y_mixed.astype(np.float32)


# ===========================================================================
# 5. CORE LOGIC
# ===========================================================================

def load_images_from_single_dataset(dataset_path, image_size, dataset_name):
    """
    Loads all images from a dataset directory structured with class subfolders.

    Args:
        dataset_path (str): The path to the root dataset directory.
        image_size (tuple): The target (width, height) to resize images to.
        dataset_name (str): The name of the dataset (for logging).

    Returns:
        tuple: A tuple containing:
            - (np.ndarray): An array of all loaded and resized images.
            - (np.ndarray): An array of corresponding string labels.
    """
    images, labels = [], []
    
    if not os.path.isdir(dataset_path):
        print_error(f"Data directory not found after check: {dataset_path}")
        return np.array(images), np.array(labels)
        
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    if not class_names:
        print_error(f"No class subdirectories found in: {dataset_path}")
        return np.array(images), np.array(labels)
        
    for class_name in tqdm(class_names, desc=f"   {Style.GREEN}Loading {dataset_name}{Style.RESET}", ncols=100, bar_format='{l_bar}{bar}|'):
        class_path = os.path.join(dataset_path, class_name)
        for image_name in os.listdir(class_path):
            try:
                img = cv2.imread(os.path.join(class_path, image_name))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(class_name)
            except Exception:
                continue
    return np.array(images), np.array(labels)


def build_and_finetune_model(X_train_img, y_train_enc, X_val_img, y_val_enc, epochs):
    """
    Builds and fine-tunes a ConvNeXtBase model for feature extraction.

    This function sets up the ConvNeXtBase model with a custom head,
    unfreezes the last 12 layers for aggressive fine-tuning, and trains
    it using a custom generator that applies Mixup augmentation.

    Args:
        X_train_img (np.ndarray): Training images.
        y_train_enc (np.ndarray): Integer-encoded training labels.
        X_val_img (np.ndarray): Validation images.
        y_val_enc (np.ndarray): Integer-encoded validation labels.
        epochs (int): The maximum number of epochs for fine-tuning.

    Returns:
        tensorflow.keras.models.Model: The trained feature extractor model,
                                       with output from the 'feature_layer'.
    """
    print_info(f"Using feature extractor: {Style.BOLD}ConvNeXtBase{Style.RESET} with {Style.BOLD}Mixup Augmentation{Style.RESET}.")
    
    base_model = ConvNeXtBase(
        weights='imagenet', 
        include_top=False, 
        include_preprocessing=True, 
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers: layer.trainable = False
    
    unfreeze_count = 12
    print_info(f"Applying aggressive fine-tuning: Unfreezing the last {Style.BOLD}{unfreeze_count}{Style.RESET} layers.")
    for layer in base_model.layers[-unfreeze_count:]: layer.trainable = True 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    print_info(f"Applying new simplified head: {Style.BOLD}GAP -> Dense(512) -> BN -> Dropout{Style.RESET}")
    feature_output = Dense(512, activation='relu', name='feature_layer', kernel_regularizer=l2(Config.L2_REG_FACTOR))(x)
    
    x_for_prediction = BatchNormalization()(feature_output)
    x_for_prediction = Dropout(0.5)(x_for_prediction)
    
    num_classes = len(np.unique(np.concatenate((y_train_enc, y_val_enc))))
    prediction_layer = Dense(num_classes, activation='softmax')(x_for_prediction)
    
    model = Model(inputs=base_model.input, outputs=prediction_layer)
    
    # LOSS CHANGE FOR MIXUP: Use categorical_crossentropy for soft labels
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print_info(f"Unleashing the Neural Network for {Style.BOLD}{epochs}{Style.RESET} epochs (max)...")
    
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    
    y_val_one_hot = to_categorical(y_val_enc, num_classes=num_classes)
    
    train_generator = datagen.flow(X_train_img, y_train_enc, batch_size=Config.FINETUNE_BATCH_SIZE, shuffle=True)
    
    def mixup_generator():
        """
        A generator that applies geometric augmentation and then Mixup.
        
        Yields:
            tuple: A batch of (X_mixed, y_mixed) data.
        """
        for X_geom, y_int in train_generator:
            X_mixed, y_mixed = mixup_data(X_geom, y_int, num_classes)
            yield X_mixed, y_mixed

    train_dataset = tf.data.Dataset.from_generator(
        mixup_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *Config.IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32) 
        )
    ).repeat().prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(X_train_img) // Config.FINETUNE_BATCH_SIZE

    early_stopper = EarlyStopping(monitor='val_loss', patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val_img, y_val_one_hot),
        callbacks=[early_stopper],
        verbose=1
    )
    
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer('feature_layer').output)
    feature_extractor.compile(optimizer='adam', loss='mse') 
    return feature_extractor


class GreyWolfOptimizer:
    """
    Implements the Grey Wolf Optimizer (GWO) for feature selection.
    
    Attributes:
        n_wolves (int): Number of search agents (wolves).
        max_iter (int): Maximum number of iterations.
        dim (int): The dimensionality of the search space (total features).
    """
    def __init__(self, n_wolves, max_iter, dim):
        """
        Initializes the GreyWolfOptimizer.

        Args:
            n_wolves (int): Number of search agents.
            max_iter (int): Maximum number of iterations.
            dim (int): Dimensionality (number of features).
        """
        self.n_wolves, self.max_iter, self.dim = n_wolves, max_iter, dim

    def _fitness(self, pos, X_tr, y_tr, X_v, y_v):
        """
        Calculates the fitness of a single wolf (solution).

        Fitness is a weighted combination of the validation error (high weight)
        and the feature reduction rate (low weight).

        Args:
            pos (np.ndarray): The binary position vector (chromosome) of the wolf.
            X_tr (np.ndarray): Scaled training features.
            y_tr (np.ndarray): Training labels.
            X_v (np.ndarray): Scaled validation features.
            y_v (np.ndarray): Validation labels.

        Returns:
            float: The calculated fitness score (lower is better).
        """
        selected = np.where(pos == 1)[0]
        if len(selected) == 0: return 1.0
        svm = SVC(kernel=Config.SVM_KERNEL, C=Config.SVM_C, random_state=Config.RANDOM_STATE).fit(X_tr[:, selected], y_tr)
        error_weight = Config.FITNESS_ERROR_WEIGHT 
        return error_weight * (1 - accuracy_score(y_v, svm.predict(X_v[:, selected]))) + (1 - error_weight) * (len(selected) / self.dim)

    def optimize(self, X_train, y_train, X_val, y_val):
        """
        Runs the GWO optimization loop.

        Args:
            X_train (np.ndarray): Scaled training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Scaled validation features.
            y_val (np.ndarray): Validation labels.

        Returns:
            np.ndarray: The best position vector (alpha_pos) found.
        """
        pos = np.random.randint(0, 2, (self.n_wolves, self.dim))
        alpha_pos, beta_pos, delta_pos = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')

        for it in tqdm(range(self.max_iter), desc=f"   {Style.BOLD}🐺 GWO Hunting...{Style.RESET}  ", ncols=100, bar_format='{l_bar}{bar}|'):
            for i in range(self.n_wolves):
                fit = self._fitness(pos[i], X_train, y_train, X_val, y_val)
                if fit < alpha_score: alpha_score, alpha_pos = fit, pos[i].copy()
                elif alpha_score < fit < beta_score: beta_score, beta_pos = fit, pos[i].copy()
                elif beta_score < fit < delta_score: delta_score, delta_pos = fit, pos[i].copy()
            a = 2 - it * (2 / self.max_iter)
            for i in range(self.n_wolves):
                A1, A2, A3 = a * (2 * np.random.rand() - 1), a * (2 * np.random.rand() - 1), a * (2 * np.random.rand() - 1)
                X1, X2, X3 = alpha_pos - A1 * np.abs(2 * np.random.rand() * alpha_pos - pos[i]), beta_pos - A2 * np.abs(2 * np.random.rand() * beta_pos - pos[i]), delta_pos - A3 * np.abs(2 * np.random.rand() * delta_pos - pos[i])
                pos[i] = (1 / (1 + np.exp(-((X1 + X2 + X3) / 3))) > np.random.rand(self.dim)).astype(int)
        return alpha_pos


class GeneticAlgorithm:
    """
    Implements a Genetic Algorithm (GA) for feature selection.

    Attributes:
        pop_size (int): The size of the population.
        gens (int): The number of generations to run.
        dim (int): The dimensionality of the search space (total features).
        mut_rate (float): The mutation rate.
    """
    def __init__(self, pop_size, gens, dim, mut_rate=0.1):
        """
        Initializes the GeneticAlgorithm.

        Args:
            pop_size (int): Population size.
            gens (int): Number of generations.
            dim (int): Dimensionality (number of features).
            mut_rate (float, optional): Mutation rate. Defaults to 0.1.
        """
        self.pop_size, self.gens, self.dim, self.mut_rate = pop_size, gens, dim, mut_rate

    def _fitness(self, chrom, X_tr, y_tr, X_v, y_v):
        """
        Calculates the fitness of a single chromosome.

        Args:
            chrom (np.ndarray): The binary chromosome (solution).
            X_tr (np.ndarray): Scaled training features.
            y_tr (np.ndarray): Training labels.
            X_v (np.ndarray): Scaled validation features.
            y_v (np.ndarray): Validation labels.

        Returns:
            float: The calculated fitness score (lower is better).
        """
        # Calls the same fitness function used by GWO for consistency
        return GreyWolfOptimizer(1, 1, self.dim)._fitness(chrom, X_tr, y_tr, X_v, y_v)

    def optimize(self, X_train, y_train, X_val, y_val, initial_pop=None):
        """
        Runs the GA optimization loop.

        Args:
            X_train (np.ndarray): Scaled training features.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Scaled validation features.
            y_val (np.ndarray): Validation labels.
            initial_pop (np.ndarray, optional): An initial population to start
                                                with (e.g., from GWO). 
                                                Defaults to None (random).

        Returns:
            np.ndarray: The best chromosome found.
        """
        pop = initial_pop if initial_pop is not None else np.random.randint(0, 2, (self.pop_size, self.dim))
        best_chrom, best_fit = None, float('inf')

        for gen in tqdm(range(self.gens), desc=f"   {Style.BOLD}🧬 GA Evolving...{Style.RESET} ", ncols=100, bar_format='{l_bar}{bar}|'):
            fits = [self._fitness(c, X_train, y_train, X_val, y_val) for c in pop]
            best_idx = np.argmin(fits)
            if fits[best_idx] < best_fit: best_fit, best_chrom = fits[best_idx], pop[best_idx]
            
            selected_indices = np.random.randint(0, self.pop_size, self.pop_size * 2)
            selected = [pop[i] if fits[i] < fits[j] else pop[j] for i, j in zip(selected_indices[::2], selected_indices[1::2])]
            
            new_pop = []
            for i in range(0, len(selected), 2):
                p1, p2 = selected[i], selected[i+1]
                c_point = np.random.randint(1, self.dim)
                c1, c2 = np.concatenate([p1[:c_point], p2[c_point:]]), np.concatenate([p2[:c_point], p1[c_point:]])
                for chrom in [c1, c2]:
                    if np.random.rand() < Config.GA_MUTATION_RATE: chrom[np.random.randint(0, self.dim)] ^= 1
                    new_pop.append(chrom)
            pop = np.array(new_pop)
        return best_chrom

# ===========================================================================
# 6. SINGLE DATASET PIPELINE
# ===========================================================================
def process_single_dataset(dataset_name, dataset_config):
    """
    Executes the full analysis pipeline for one dataset.

    This includes:
    1. Unzipping and loading data.
    2. Fine-tuning ConvNeXt for feature extraction (or loading from cache).
    3. Running GWO, then GA (GWOGA) for feature selection.
    4. Training a final SVM on the selected features.
    5. Printing a final classification report and confusion matrix.

    Args:
        dataset_name (str): The key of the dataset from Config.DATASET_CATALOG.
        dataset_config (dict): The configuration dictionary for that dataset.
    """
    start_time = time.time()
    print_header(f"🚀 INITIATING ANALYSIS FOR: {dataset_name} 🚀")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    sel_cfg = Config.DATASET_CATALOG[dataset_name]
    
    dataset_path = auto_unzip_and_get_path(Config.BASE_DATA_DIR, sel_cfg['zip_name'], sel_cfg['image_path_relative'])
    
    if dataset_path is None:
        print_error(f"Skipping {dataset_name} due to path/zip error.")
        return

    features_cache_path = os.path.join(Config.OUTPUT_DIR, f'{dataset_name}_split_features.pkl')
    final_model_path = os.path.join(Config.OUTPUT_DIR, f'{dataset_name}_final_model.pkl')

    print_info(f"Using custom settings: {Style.BOLD}{dataset_config}{Style.RESET}")

    print_header("PHASE 1: Data Assimilation & Feature Forging (ConvNeXt)", color=Style.MAGENTA)
    if os.path.exists(features_cache_path):
        print_info(f"Found cached features. Loading from the vault...")
        with open(features_cache_path, 'rb') as f:
            features_data = pickle.load(f)
        
        features_train = features_data['X_tr']
        y_train_enc = features_data['y_tr']
        features_test = features_data['X_te']
        y_test_enc = features_data['y_te']
        labels = features_data['labels']
        print_success("Cached features successfully loaded.")
    else:
        print_info("No cached features found. Time to forge new ones with ConvNeXt.")
        images, labels = load_images_from_single_dataset(dataset_path, Config.IMAGE_SIZE, dataset_name)
        if len(images) == 0:
            print_error(f"No images found for dataset {dataset_name} in {dataset_path}. Skipping.")
            return

        le = LabelEncoder().fit(labels)
        encoded_labels = le.transform(labels)

        # This X_test_img/y_test_enc is the FINAL holdout set.
        print_info(f"Creating master train/test split ({1.0 - Config.TEST_SIZE:.0%}/{Config.TEST_SIZE:.0%})...")
        X_train_img, X_test_img, y_train_enc, y_test_enc = train_test_split(
            images, encoded_labels, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE, 
            stratify=encoded_labels
        )

        X_train_finetune_img, X_val_finetune_img, y_train_finetune_enc, y_val_finetune_enc = train_test_split(
            X_train_img, y_train_enc, 
            test_size=0.2, 
            random_state=Config.RANDOM_STATE, 
            stratify=y_train_enc
        )
        
        feature_extractor = build_and_finetune_model(
            X_train_finetune_img, y_train_finetune_enc, 
            X_val_finetune_img, y_val_finetune_enc, 
            epochs=dataset_config["FINETUNE_EPOCHS"]
        )
        
        print_info("Extracting conceptual features from TRAIN/VAL images...")
        features_train = feature_extractor.predict(X_train_img, verbose=1)
        
        print_info("Extracting conceptual features from final TEST (holdout) images...")
        features_test = feature_extractor.predict(X_test_img, verbose=1)
        
        print_info(f"Caching features to the vault: {features_cache_path}")
        features_data = {
            'X_tr': features_train, 'y_tr': y_train_enc,
            'X_te': features_test, 'y_te': y_test_enc,
            'labels': labels 
        }
        with open(features_cache_path, 'wb') as f: pickle.dump(features_data, f)
        print_success("Feature extraction and caching complete.")

    print_header("PHASE 2: Intelligent Feature Selection with GWOGA", color=Style.MAGENTA)
    
    X_tr_features = features_train
    y_tr_labels = y_train_enc
    X_te_features = features_test
    y_te_labels = y_test_enc

    le = LabelEncoder().fit(labels) 
    
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr_features, y_tr_labels, 
        test_size=0.1, 
        random_state=Config.RANDOM_STATE, 
        stratify=y_tr_labels
    )
    
    scaler = StandardScaler()
    
    # Fit scaler ONLY on GWO/GA training set
    X_tr = scaler.fit_transform(X_tr)
    # Transform GWO/GA validation set
    X_v = scaler.transform(X_v)
    # Transform the FINAL (holdout) test set
    X_te = scaler.transform(X_te_features)
    
    gwo_iter = sel_cfg["GWO_MAX_ITER"]
    ga_gens = sel_cfg["GA_GENERATIONS"]
    
    gwo_best = GreyWolfOptimizer(Config.GWO_N_WOLVES, gwo_iter, X_tr.shape[1]).optimize(X_tr, y_tr, X_v, y_v)
    initial_pop = np.random.randint(0, 2, (Config.GA_POPULATION_SIZE, X_tr.shape[1])); initial_pop[0] = gwo_best
    selected_features = GeneticAlgorithm(Config.GA_POPULATION_SIZE, ga_gens, X_tr.shape[1]).optimize(X_tr, y_tr, X_v, y_v, initial_pop=initial_pop)
    selected_indices = np.where(selected_features == 1)[0]

    print_header("PHASE 3: Final Model Training & Verdict", color=Style.MAGENTA)
    print_info(f"Original feature count: {X_tr.shape[1]}")
    print_info(f"Features selected by GWOGA: {len(selected_indices)}")
    if X_tr.shape[1] > 0 and len(selected_indices) > 0:
        print_info(f"Feature reduction rate: {Style.BOLD}{(1 - len(selected_indices) / X_tr.shape[1]) * 100:.2f}%{Style.RESET}")

        # Train final SVM on the scaled GWO/GA training data (X_tr, y_tr)
        final_svm = SVC(kernel=Config.SVM_KERNEL, C=Config.SVM_C, probability=True, random_state=Config.RANDOM_STATE).fit(X_tr[:, selected_indices], y_tr)
        
        # Predict on the scaled FINAL (holdout) test set (X_te)
        y_pred = final_svm.predict(X_te[:, selected_indices])
        
        print_header(f"📊 FINAL REPORT CARD FOR: {dataset_name} 📊", color=Style.GREEN)
        
        # Compare prediction against the FINAL (holdout) labels (y_te_labels)
        print(f"\n   🏆 {Style.BOLD}Overall Model Accuracy: {accuracy_score(y_te_labels, y_pred) * 100:.2f}%{Style.RESET} 🏆\n")
        
        target_names = le.classes_[np.unique(np.concatenate((y_te_labels, y_pred)))]
        print("--- Classification Report ---\n")
        print(classification_report(y_te_labels, y_pred, target_names=target_names, zero_division=0))
        print("\n--- Confusion Matrix ---\n")
        print(pd.DataFrame(confusion_matrix(y_te_labels, y_pred), index=target_names, columns=target_names))
        
        final_model_package = {'model': final_svm, 'scaler': scaler, 'label_encoder': le, 'selected_features_indices': selected_indices}
        with open(final_model_path, 'wb') as f: pickle.dump(final_model_package, f)
        print_success(f"\nFinal model for {dataset_name} secured in the vault: {final_model_path}")
    else:
        print_error(f"No features were selected. The model could not be trained.")
    
    total_time = time.time() - start_time
    print_header(f"🎉 MISSION COMPLETE for {dataset_name} in {total_time:.2f} seconds 🎉", color=Style.GREEN)

# ===========================================================================
# 7. MAIN INTERACTIVE LOOP
# ===========================================================================
def main():
    """
    Runs the main interactive command-line loop.

    Allows the user to select which dataset to analyze or to quit.
    """
    while True:
        print_header("✨ HYBRID GWO-GA: DATASET ANALYSIS CONSOLE ✨")
        datasets = list(Config.DATASET_CATALOG.keys())
        for i, name in enumerate(datasets):
            print(f"   {Style.CYAN}[{i+1}]{Style.RESET} Analyze the '{Style.BOLD}{name}{Style.RESET}' dataset")
        print(f"   {Style.YELLOW}[q]{Style.RESET} Quit Mission")
        
        choice = input(f"\n{Style.BOLD}Enter your command (1, 2, 3, or q): {Style.RESET}").strip().lower()

        if choice == 'q':
            print("\nShutting down. Goodbye! 👋")
            break
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(datasets):
                dataset_name = datasets[choice_idx]
                dataset_config = Config.DATASET_CATALOG[dataset_name]
                
                process_single_dataset(dataset_name, dataset_config)
            else:
                print_error(f"Invalid command. Please choose from the list.")
        except ValueError:
            print_error(f"Invalid command. Please enter a number or 'q'.")
        
        while True:
            another = input(f"\n{Style.BOLD}Engage another dataset? (y/n): {Style.RESET}").strip().lower()
            if another in ['y', 'n']: break
            print_error(f"Invalid input. Please enter 'y' or 'n'.")
        
        if another == 'n':
            print("\nMission concluded. Goodbye! 👋")
            break

if __name__ == "__main__":
    main()