import numpy as np
import pickle
from AdaBoost import AdaBoost
from haar_utils import *
class ViolaJones:
    def __init__(self,layers,featurespath) -> None:
        """
        layers: Array of T->(Number of weak classifiers) used in cascade classifier
        """
        self.layers = layers
        self.clfs = []
        self.width =19
        self.height = 19
        self.featurespath=featurespath
    def train(self , x ,y):
        """
        Useing booseted cascade of features , each classifier is trained
        with positive examples plus the false positive of the previous one
        """

        print("Preparing data....")

        pos_num = np.sum(y)
        neg_num = len(y)-pos_num
        img_h , img_w = x[0].shape
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        # Show data info
        print("Summary input data:")
        print("  - Total faces: "+str(int(pos_num)) + str( 100.0 * pos_num / (pos_num + neg_num))+"%")
        print("  - Total non-faces: "+ str(int(neg_num)) + str( 100.0 * neg_num / (pos_num + neg_num)))
        print("  - Total samples: "+ str(int(pos_num + neg_num)))
        print("  - Size (WxH): " + str(img_w, img_h))


        print("Generating integral images...")
        x_integ = np.array(list(map(lambda y: integeral_img(y),x)) , dtype=np.uint32)
        print("  - Num. integral images: "+ str(len(x_integ)))

        print("Building features")
        features = build_features(img_w, img_h)
        print(" - Num. features: "+ str(len(features)))
        print("Applying features...")
        done_features = self.load_applied_features(str(pos_num),str(neg_num))
        if done_features is None:
            done_features = apply_features(x_integ,features)
            np.save('applied_features' +self.featurespath+ "xf" + ".npy", done_features)
            print("Applied features file saved!")
        
        print("  - Num. features applied: " + str(len(done_features) * len(features)))
        for i,t in enumerate(self.layers):
            print("[CascadeClassifier] Training {} of out {} layers".format(i+1, len(self.layers)))
            if len(neg_indices) == 0:
                print('No Negative Samples...Stop')
                break

            mrgd_indxs = np.concatenate([pos_indices,neg_indices])
            np.random.shuffle(mrgd_indxs)
            clf = AdaBoost(T=t)
            clf.train(done_features[:,mrgd_indxs],y[mrgd_indxs],features)
            self.clfs.append(clf)

            # store non-faces where labled as a face to pass it to new stage
            fp= []
            for neg_indx in neg_indices:
                if self.classify(x[neg_indx]) == 1:
                    fp.append(neg_indx)
            neg_indices = np.array(fp)
    def classify(self, image, scale=1.0):
        """
        If a no-face is found, reject now. Else, keep looking.
        """
        return self.classify_ii(integeral_img(image), scale)

    def classify_ii(self, ii, scale=1.0):
        """
        If a no-face is found, reject now. Else, keep looking.
        """
        for clf in self.clfs:  # ViolaJones
            if clf.classify(ii, scale) == 0:
                return 0
        return 1
    
    def save(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    
    def load_applied_features(self,pos_num,neg_num):
        X_f = None
        # Load precomputed features
        try:
            X_f = np.load('applied_features' +self.featurespath+ "xf" + ".npy")
            print("Precomputed dataset loaded!")
        except FileNotFoundError:
            pass
        return X_f