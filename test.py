from NaiveBayes import *
import sys
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def read_file():
    """
    Tries to read csv file using Pandas
    """
    # Get the name of the data file and load it into 
    if len(sys.argv) < 2:
        # Ask the user for the name of the file
        print "Filename: ", 
        filename = sys.stdin.readline().strip()
    else:
        filename = sys.argv[1]

    try:
        data = pd.read_csv(filename)
    except IOError:
        print "Error: The file '%s' was not found on this system." % filename
        sys.exit(0)

    return data

def build_model(data):
    """
    This function builds tree from Pandas Data Frame with last column as the dependent feature
    """
    attributes = list(data.columns.values)
    target = attributes[-1]
    classes = get_attribute_values(data,target)
    class_frequency = data[target].value_counts().sort_index()
    betak = [2]*len(classes)
    betam = (100/class_frequency + 1).astype(int)
    return NaiveBayesMAP(data,target,betak,betam)


if __name__ == "__main__":
    data = read_file()
    del data['animalname']
    print "Data Read and Loaded"
    X_train, X_valid = train_test_split(data, test_size=0.1, random_state=15)
    model = build_model(X_train)
    actuals = X_valid.type
    del X_valid['type']
    preds = predict(model,X_valid)
    print "Classification Accuracy is " + str(accuracy_score(actuals,preds))


    
