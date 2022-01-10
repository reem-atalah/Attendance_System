import numpy as np

class WeakClassifier:
    def __init__(self,haar_feature=None , threshold=None , polarity=None) -> None:
        self.haar_feature = haar_feature
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, ii , scale=1.0):
        """
        Classifies an image given its integral image
        """
        feature_value = self.haar_feature.get_haar_feature_value(ii,scale)
        return 1 if self.polarity * feature_value < self.polarity * self.threshold * (scale**2) else 0
    def classify_with_feature(self , feature_value):
        """
        Classifies an image given its feature value
        """    
        a = self.polarity * feature_value
        b = self.polarity * self.threshold
        return np.less(a,b).astype(int)
    def train(self , x , y , weights , total_pos,total_neg):
        # Sort features
        sorted_features = sorted(zip(weights, x, y), key=lambda a: a[1])
        pos_seen, neg_seen = 0, 0
        pos_weights, neg_weights = 0, 0
        min_error = float('inf')
        for w,f,label in sorted_features:
            # Calculate error
            error=min(neg_weights + (total_pos - pos_weights), pos_weights + (total_neg - neg_weights))
            
            if error < min_error:
                min_error = error
                self.threshold = f
                self.polarity = 1 if pos_seen > neg_seen else -1
            
            if label == 1:
                pos_seen += 1
                pos_weights += w
            else:
                neg_seen += 1
                neg_weights += w