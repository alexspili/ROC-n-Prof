from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox, layout, gridplot
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import Select, TextInput, Slider, DataTable, DateFormatter, TableColumn, IntEditor
import numpy as np
from numpy import random

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

TPprofit_init=1000
FPprofit_init=-1000
TNprofit_init=0
FNprofit_init=-100
threshold_init=50


def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list

    Takes numpy arrays of the predicted probabilities and true labels.
    Returns the True Positive Rates and False Positive Rates for the ROC curve.

    Calls function confusion_matrix
    '''
    TPrate = []
    FPrate = []
    thresholds = np.unique(np.sort(probabilities))
    thresholds = thresholds[thresholds >= 0]
    for threshold in thresholds:
        d = confusion_matrix(probabilities, labels, threshold)
        TPrate.append(d["TP"]/float(d["TP"]+d["FN"]))
        FPrate.append(d["FP"]/float(d["FP"]+d["TN"]))
    return TPrate, FPrate

def confusion_matrix(probabilities, labels, threshold):
    '''
    INPUT: numpy array, numpy array, float in [0,1]
    OUTPUT: dictionary

    Takes numpy arrays of the predicted probabilities and true labels and a Threshold value
    Returns a dictionary with the True Positive, False Positive, True Negative, False Negative counts
    '''
    FN = 0
    FP = 0
    TN = 0
    TP = 0
    for i, j in zip(probabilities, labels):
        if j == 0 and i > threshold:
            FP += 1
        elif j == 0 and i <= threshold:
            TN += 1
        elif j == 1 and i > threshold:
            TP += 1
        elif j == 1 and i <= threshold:
            FN += 1
    return {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

def profit_curve(probabilities, labels, dict_benefits):
    '''
    INPUT: numpy array, numpy array, dictionary
    OUTPUT: list, list

    Takes numpy arrays of the predicted probabilities and true labels and the
    dictionary of float values for TPprofit, FPprofit, FNprofit, TNprofit
    Returns the Profit values and Thresholds for the ROC curve.

    Calls functions confusion_matrix, get_profit
    '''
    profit_list = []
    x_points = 101
    thresholds = np.linspace(0,1,x_points)
    for threshold in thresholds:
        dict_counts = confusion_matrix(probabilities, labels, threshold)
        profit = get_profit(dict_counts, dict_benefits)
        profit_list.append(profit)
    return profit_list, thresholds*100

def get_profit(dict_counts, dict_benefits):
    '''
    INPUT: dictionary, dictionary
    OUTPUT: float

    Takes the dictionary of int counts of TP, FP, FN, TN and the dictionary of float values for TPprofit, FPprofit, FNprofit, TNprofit
    Returns the profit per case (= total profit divided by total count)
    '''
    s = sum(dict_counts.values())
    return sum([dict_counts[i]*dict_benefits[i] for i in ["TP", "FP", "FN", "TN"] ] ) / float(s)

def update_slider(attrname, old, new):
    '''
    The callback function for the threshold value slider.
    Updates the confusion matrix and the position of the threshold on the ROC and profit graphs
    '''
    threshold=slider_threshold.value
    d = confusion_matrix(probabilities, y_test, threshold/100.)

    confusion_matrix_source.data = dict(
            pred_Y=['TP = '+str(d["TP"]),'FP = '+str(d["FP"]), 'SUM = '+str(d["FP"]+d["TP"])],
            pred_N=['FN = '+str(d["FN"]),'TN = '+str(d["TN"]), 'SUM = '+str(d["FN"]+d["TN"])],
        )

    ROC_dot_source.data = dict(x=[d["FP"]/float(d["TN"]+d["FP"])], y=[d["TP"]/float(d["TP"]+d["FN"])])

    profit_dot_source.data = dict(x=[threshold], y=[ get_profit(d, get_benefits_dict() ) ])

def update_profit_textboxes(attrname, old, new):
    '''
    The callback function for the benefit text boxes.
    Updates the profit curve
    '''
    profits, thresholds = profit_curve(probabilities, y_test,  get_benefits_dict() )
    profit_line_source.data = dict(x=thresholds, y=profits)

    threshold=slider_threshold.value
    dict_counts = confusion_matrix(probabilities, y_test, threshold/100.) # these values revert to the initial ones; alternatively, make them global
    profit_dot_source.data = dict(x=[threshold], y=[ get_profit(dict_counts, get_benefits_dict() )])

def get_benefits_dict():
    return {'TP': float(text_TP.value), 'FP': float(text_FP.value), 'FN': float(text_FN.value), 'TN': float(text_TN.value)}

#LOAD DATA AND RUN MODEL
df = pd.read_csv('loanf.csv')
y = (df['Interest.Rate'] <= 12).values
X = df[['FICO.Score', 'Loan.Length', 'Loan.Amount']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99, test_size=0.4)
model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]

# GET INITIAL VALUES
d = confusion_matrix(probabilities, y_test, threshold_init/100.)

# BENEFIT BOXES
text_TP = TextInput(value=str(TPprofit_init), title="Benefit TP:")
text_FP = TextInput(value=str(FPprofit_init), title="Benefit FP:")
text_TN = TextInput(value=str(TNprofit_init), title="Benefit TN:")
text_FN = TextInput(value=str(FNprofit_init), title="Benefit FN:")

# SLIDER
slider_threshold = Slider(start=0, end=100, step=1, value=threshold_init, title="Threshold [%]")

# ROC CURVE
p1 = Figure(title="ROC curve", plot_height=500, plot_width=500, x_range=(0,1), y_range=(0,1))
p1.xaxis.axis_label = '1 - Specificity = FP/(FP+TN)'
p1.yaxis.axis_label = 'Sensitivity = TP/(TP+FN)'

TPrate, FPrate = roc_curve(probabilities, y_test)
ROC_line_source = ColumnDataSource(data=dict(x=FPrate, y=TPrate))
p1.line(x='x', y='y', source=ROC_line_source, color="#2222aa", line_width=1)

ROC_dot_source = ColumnDataSource(data=dict(x=[d["FP"]/float(d["TN"]+d["FP"])], y=[d["TP"]/float(d["TP"]+d["FN"])]) )
p1.circle(x='x', y='y', size=10, fill_color='red', source=ROC_dot_source)

# CONFUSION MATRIX
data = dict(
        pred_Y=['TP = '+str(d["TP"]),'FP = '+str(d["FP"]), 'SUM = '+str(d["FP"]+d["TP"])],
        pred_N=['FN = '+str(d["FN"]),'TN = '+str(d["TN"]), 'SUM = '+str(d["FN"]+d["TN"])],
    )

confusion_matrix_source = ColumnDataSource(data)

columns_confusion = [
        TableColumn(field="pred_Y", title="Predicted Positive"),
        TableColumn(field="pred_N", title="Predicted Negative"),
    ]
confusion_table = DataTable(source=confusion_matrix_source, columns=columns_confusion, sortable=False, width=400, height=100)

# PROFIT CURVE
p2 = Figure(title="Profit curve", plot_height=500, plot_width=500, x_range=(0,100))
p2.xaxis.axis_label = 'Threshold [%]'
p2.yaxis.axis_label = 'Profit per case'

profits_profitC, thresholds_profitC = profit_curve(probabilities, y_test, get_benefits_dict() )
profit_line_source = ColumnDataSource(data=dict(x=thresholds_profitC, y=profits_profitC))
p2.line(x='x', y='y', source=profit_line_source, color="#2222aa", line_width=1)

profit_dot_source = ColumnDataSource(data=dict(x=[threshold_init], y=[get_profit(d, get_benefits_dict() ) ]))
p2.circle(x='x', y='y', size=10, fill_color='red', source=profit_dot_source)

# WIDGETS CALLBACKS
for textBox in [text_TP, text_FP, text_TN, text_FN]:
    textBox.on_change('value', update_profit_textboxes)

slider_threshold.on_change('value', update_slider)

# LAYOUT - CAN'T GET RID OF THE BOKEH LOGO HERE
grid = gridplot([[text_TP, text_FN], [text_FP, text_TN]], logo=None)

l = row( column(p1,  slider_threshold, confusion_table ), column(p2, grid) )
curdoc().add_root(l)
