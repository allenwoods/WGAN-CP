import numpy as np
from scipy.stats import entropy

def get_inception_score(img_batch_arr, evaluator, splits=10):
    batch_size = 100
    num_batches = int(np.ceil(float(len(img_batch_arr)) / float(batch_size)))
    preds = [evaluator.predict(img_chunk) for img_chunk in np.array_split(img_batch_arr, num_batches)]
    preds = np.concatenate(preds, 0)  # Predictive labels of images
    mean_confidence = np.mean(np.max(preds, axis=1))
    class_entropy = entropy(np.bincount(np.argmax(preds, axis=1), minlength=10)/len(img_batch_arr))
    scores = []
    for i in range(splits):
        pred = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = pred * (np.log(pred) - np.log(np.expand_dims(np.mean(pred, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return mean_confidence, class_entropy, np.mean(scores), np.std(scores)
