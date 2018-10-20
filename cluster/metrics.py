OUTLIER_LABEL = -1

def purity(predict_labels, expected_labels):
    unique_prediction_labels = list(set(predict_labels))
    purity = 0
    for label in unique_prediction_labels:
        members_dict = {}
        for i in range(len(predict_labels)):
            if (predict_labels[i] == label):
                if (expected_labels[i] in members_dict):
                    members_dict[expected_labels[i]] += 1
                else:
                    members_dict[expected_labels[i]] = 1
        purity += find_mode(members_dict)
    return purity/len(predict_labels)

def find_mode(dict):
    max = 0
    for key, val in dict.items():
        if (key != OUTLIER_LABEL and max < val):
            max = val
    return max