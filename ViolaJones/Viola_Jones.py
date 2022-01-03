#!/usr/bin/env python
# coding: utf-8

# In[5]:



from HaarLikeFeatures import *
#from sklearn.feature_selection import SelectPercentile, f_classif
import math
import pickle
import concurrent.futures
import tqdm
# In[7]:


class ViolaJones():
    def __init__(self, T = 50):
        """
         T: The number of weak classifiers which should be used
        """
        self.T = T
        # Inorder to save alpha values
        self.alphas = []
        self.clfs = []

    def pre_train(self, training_set , pos_num , neg_num):
        weights = np.zeros(len(training_set))
        # get the integral image of training set
        training_set_integ = []
        for img in range (len(training_set)):
            # Assuming that the traning set are array of 2D tuples (img , pos=1/neg=0)
            training_set_integ.append((integeral_img(training_set[img][0]) , training_set[img][1]))
            # Initializing the weights
            if training_set[img][1] == 1 :
                weights[img] = 1.0/(2*pos_num)
            else:
                weights[img] = 1.0/(2*neg_num)
        print("Building features")
        count ,features = HaarlikeFeatureDetector.determineFeatures( training_set_integ[0][0].shape[0] ,training_set_integ[0][0].shape[1])
        print("Applying features to training examples")
        X, y = HaarlikeFeatureDetector.apply_features(features,  training_set_integ)
        X_len  = len(X)
        Y_len = len(y)
        print("Len X" + str(X_len))
        print("Len Y" + str(Y_len))
        return X,y,X_len,training_set_integ,weights,Y_len
    def train(self, X,y,X_len,training_set_integ,weights,Y_len,features) -> None:
        """
        training_set : is the images that the model will train on it
        pos_num  : number of images containig faces
        neg_num : number of images that does not contain face

        """
        print("Selected %d potential features" % len(X))
        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            results_processes = []
            with concurrent.futures.ProcessPoolExecutor() as ex:
                train_weak_result = [ex.submit(
                    self.train_weak, X[int(X_len/10)*i:int(X_len/10)*(i+1)], y[int(Y_len/10)*i:int(Y_len/10)*(i+1)], features, weights,training_set_integ) for i in range(10)]
                for f in concurrent.futures.as_completed(train_weak_result):
                    results_processes.append(f.result())
            clf, error, accuracy = self.select_best(results_processes, weights, training_set_integ)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))

    def train_weak(self , x ,y , features , weights,training_set_integ) -> list :
        """
        Determines the optimal threshold for weak classifier 
        Each Haar-Like feature represents a weak classifier (as standalone)
        X: A numpy array Each row represents the value of a single feature for each training example (len(features), len(training_data))
        y : represents the ith training example
        features : Haar-Like features
        weights : the weight associated to each training example
        """
        total_pos , total_neg = 0.0,0.0
        # sum all weights of pos and neg lables
        # Notes: zip combines the two things into one tuple
        for w,label in zip(weights,y):
            if label == 1 :
                total_pos +=w
            else:
                total_neg +=w
        
        classifiers = []
        # Each feature represents a classifier 
        total_features = x.shape[0]
        # Note: enumerate adds a counter to iterable
        for index,feature in enumerate(x):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))
            # Sort the weights according to the feature value that they correspond to.
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_count , neg_count = 0 , 0
            pos_weights, neg_weights = 0, 0
            min_error , best_feature , best_thd , best_polarity = float('inf') , None , None , None
            for w , f , label in applied_feature :
                # this error compares how many examples will be misclassified if all examples below the current location are labeled as negative with how many examples will be misclassified if all examples below the current location are labeled as positive 
                error = min (neg_weights+total_pos - pos_weights , pos_weights+total_neg-neg_weights)
                if error < min_error :
                    min_error = error
                    best_feature = features[index]
                    best_thd = f
                    best_polarity = 1 if pos_count > neg_count else -1
                
                if label == 1:
                    pos_count += 1
                    pos_weights += w
                else:
                    neg_count += 1
                    neg_weights += w
            w_c = WeakClassifier(best_feature , best_thd , best_polarity)
            classifiers.append(w_c)
        
        return self.select_best(classifiers, weights, training_set_integ)[0]

    # Once we have trained all of the weak classifiers, we can now find the best one
    def select_best(self , classifiers , weights , training_data) : 
        """
        Selects the best classifiers based on average weighted errors
        The error equation you can find it at page 4 of original paper point 2
        classifiers: Array of weak classifiers (generate by train_weak)
        weights: the weight associated to each training example
        training_data : Training data set
        """
        best_classifier, best_error , best_accuracy = None , float('inf') , None
        # We will use all classifiers against the whole training set in order to find the best one
        for clf in classifiers :
            error , accuracy = 0 , []
            for data , w in zip (training_data , weights):
                accu = abs(clf.classify(data[0])-data[1])
                accuracy.append(accu)
                error += w*accu
            error = error / len(training_data)
            if error < best_error :
                best_classifier , best_error , best_accuracy = clf , error , accuracy
        
        return best_classifier , best_error , best_accuracy
    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = integeral_img(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0
    
    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)
        

