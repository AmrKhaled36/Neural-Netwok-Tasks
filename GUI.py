import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import matplotlib.pyplot as plt
import numpy as np
import perceptron as per
import adaline as ada
import pandas as pd
import data_loader

class window:
    
    def __init__(self, root,df):
        """
        Initialize the window.

        Parameters
        ----------
        root : tkinter.Tk
            The root window.
        df : pandas.DataFrame
            The dataframe to use.
        """
        self.root = root
        self.root.title("Task 1 NN")
        self.df = df
        self.perceptron = None
        self.adaline = None
        for i in range(5):
            self.root.grid_rowconfigure(i, weight=1)
        for i in range(4):
            self.root.grid_columnconfigure(i, weight=1)

        self.features = ["Gender", "Body mass", "Beak length", "Beak depth","Fin length"]
        self.classes = ["A", "B", "C"]
        ##First row of combo boxes.
        self.feature1 = ttk.Combobox(self.root, values=self.features, state="readonly")
        self.feature1.grid(row=0, column=0, padx=10, pady=10)
        self.feature1.set("Select feature 1")
        self.feature1.bind("<<ComboboxSelected>>", self.selected_features_classes)

        self.feature2 = ttk.Combobox(self.root, values=self.features, state="readonly")
        self.feature2.grid(row=0, column=1, padx=10, pady=10)
        self.feature2.set("Select feature 2")
        self.feature2.bind("<<ComboboxSelected>>", self.selected_features_classes)

        self.plot_button = tk.Button(self.root, text="Plot", command=self.plot, width=10)
        self.plot_button.grid(row=0, column=3, padx=10, pady=10)


        ##Second row of combo boxes.
        self.class1 = ttk.Combobox(self.root, values=self.classes, state="readonly")
        self.class1.grid(row=1, column=0, padx=10, pady=10)
        self.class1.set("Select class 1")
        self.class1.bind("<<ComboboxSelected>>", self.selected_features_classes)

        self.class2 = ttk.Combobox(self.root, values=self.classes, state="readonly")
        self.class2.grid(row=1, column=1, padx=10, pady=10)
        self.class2.set("Select class 2")
        self.class2.bind("<<ComboboxSelected>>", self.selected_features_classes)

        #Third row text boxes
        self.learning_rate_label = tk.Label(self.root, text="Learning rate:")
        self.learning_rate_label.grid(row=2, column=0, padx=10, pady=10)

        self.epochs_label = tk.Label(self.root, text="Epochs:")
        self.epochs_label.grid(row=2, column=1, padx=10, pady=10)

        self.mse_label = tk.Label(self.root, text="MSE threshold:")
        self.mse_label.grid(row=2, column=2, padx=10, pady=10)

        ##fourth row entries
        self.learning_rate_entry = tk.Entry(self.root)
        self.learning_rate_entry.grid(row=3, column=0, padx=10, pady=10)

        self.epochs_entry = tk.Entry(self.root)
        self.epochs_entry.grid(row=3, column=1, padx=10, pady=10)

        self.mse_entry = tk.Entry(self.root)
        self.mse_entry.grid(row=3, column=2, padx=10, pady=10)

        ##fifth row
        self.bias_var = tk.BooleanVar()
        self.bias_check = tk.Checkbutton(self.root, text="Bias", variable=self.bias_var)
        self.bias_check.grid(row=4, column=0, padx=10, pady=10)

        self.train_button = tk.Button(self.root, text="Train", command=self.train, width=10)
        self.train_button.grid(row=4, column=1, padx=10, pady=10)

        self.test_button = tk.Button(self.root, text="Test", command=self.test, width=10)
        self.test_button.grid(row=4, column=2, padx=10, pady=10)

        #ٌRadio buttons
        self.radio_var = tk.StringVar()
        self.radio_var.set("perceptron")

        self.radio_percep = tk.Radiobutton(self.root, text="Perceptron", variable=self.radio_var, value="perceptron")
        self.radio_percep.grid(row=0, column=2, padx=10, pady=10)

        self.radio_adal = tk.Radiobutton(self.root, text="Adaline", variable=self.radio_var, value="adaline")
        self.radio_adal.grid(row=1, column=2, padx=10, pady=10)

    def selected_features_classes(self,event):
        """
        Handle the event when the same feature or class is selected.

        Parameters
        ----------
        event : tkinter.Event
            The event that triggered the function.

        """
        if self.feature1.get() == self.feature2.get():
            messagebox.showerror("Error", "Please select different features")
            self.feature1.set("Select feature 1")
            self.feature2.set("Select feature 2")
        if self.class1.get() == self.class2.get():
            messagebox.showerror("Error", "Please select different classes")
            self.class1.set("Select class 1")
            self.class2.set("Select class 2")

    def train(self):
        """
        Function to call the train method and make sure all fields are filled.
        """
        if self.feature1.get() == "Select feature 1":
            messagebox.showerror("Error", "Please select feature 1")
            return
        if self.feature2.get() == "Select feature 2":
            messagebox.showerror("Error", "Please select feature 2")
            return
        if self.class1.get() == "Select class 1":
            messagebox.showerror("Error", "Please select class 1")
            return
        if self.class2.get() == "Select class 2":
            messagebox.showerror("Error", "Please select class 2")
            return
        if self.learning_rate_entry.get() == "":
            messagebox.showerror("Error", "Please enter learning rate")
            return
        if self.epochs_entry.get() == "":
            messagebox.showerror("Error", "Please enter epochs")
            return
        if self.mse_entry.get() == "":
            messagebox.showerror("Error", "Please enter MSE threshold")
            return
        if self.radio_var.get() == "perceptron":
            self.train_perceptron()
        else:
            self.train_adaline()

    def test(self):
        """
        Function to call the test method.
        """
        if self.radio_var.get() == "perceptron":
            self.test_perceptron()
        else:
            self.test_adaline()
        
    def selected_feature(self, feature):
        """
        Return the column name of the feature.
        """
        if feature == "Gender":
            return "gender"
        elif feature == "Body mass":
            return "body_mass"
        elif feature == "Beak length":
            return "beak_length"
        elif feature == "Beak depth":
            return "beak_depth"
        elif feature == "Fin length":
            return "fin_length"

    def dataframe_to_xy(self, df, f1, f2, class1, class2):
        """
        Return the x, y, x_train, y_train, x_test, y_test.
        """
        x, y, x_train, y_train, x_test, y_test = data_loader.load_x_y(df, f1, f2, class1, class2)
        return x, y, x_train, y_train, x_test, y_test

    def train_perceptron(self):
        """
        Train the perceptron and calculate the confusion matrix.
        """
        f1 = self.selected_feature(self.feature1.get())
        f2 = self.selected_feature(self.feature2.get())
        class1 = self.class1.get()
        class2 = self.class2.get()
        lr = float(self.learning_rate_entry.get())
        epochs = int(self.epochs_entry.get())

        self.perceptron = per.perceptron(2, lr, bias=self.bias_var.get())
        x, y, x_train, y_train, x_test, y_test = self.dataframe_to_xy(self.df, f1, f2, class1, class2)

        self.perceptron.fit(x_train, y_train, epochs)

        w1, w2, b = self.perceptron.get_weights()
        x1 = x_train[:,0]
        x2 = -(w1/w2) * x1 - (b/w2)
        pred = self.perceptron.prediction(x_train)
        cm = self.confusion_matrix(y_train, pred)
        print(f"Confusion Matrix: {cm['matrix']} \n Accuracy: {cm['accuracy']} Precision: {cm['precision']} recall: {cm['recall']} f1: {cm['f1']}")
        
        #plot the data
        plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='blue', label=class1)
        plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='orange', label=class2)

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f'{f1} vs {f2} for {class1} and {class2}')
        plt.legend(title="Classes")
        plt.plot(x1, x2, c='red')        
        plt.show()

    def train_adaline(self):
        """
        Train the adaline and calculate the confusion matrix.
        """
        f1 = self.selected_feature(self.feature1.get())
        f2 = self.selected_feature(self.feature2.get())
        class1 = self.class1.get()
        class2 = self.class2.get()
        lr = float(self.learning_rate_entry.get())
        epochs = int(self.epochs_entry.get())
        mse_threshold = float(self.mse_entry.get())

        self.adaline = ada.adaline(2, lr, bias=self.bias_var.get())
        x, y, x_train, y_train, x_test, y_test = self.dataframe_to_xy(self.df, f1, f2, class1, class2)

        self.adaline.fit(x_train, y_train, epochs, mse_threshold)

        w1, w2, b = self.adaline.get_weights()
        x1 = x_train[:,0]
        x2 = (0.5 - w1 * x1 - b) * (1/w2)
        pred = np.array(self.adaline.prediction(x_train))

        cm = self.confusion_matrix(y_train, pred)
        print(f"Confusion Matrix: {cm['matrix']} \n Accuracy: {cm['accuracy']} Precision: {cm['precision']} recall: {cm['recall']} f1: {cm['f1']}")
        
        #plot the data
        plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='blue', label=class1)
        plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='orange', label=class2)

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f'{f1} vs {f2} for {class1} and {class2}')
        plt.legend(title="Classes")
        plt.plot(x1, x2, c='red')        
        plt.show()

    def test_perceptron(self):
        """
        Test the perceptron and calculate the confusion matrix.
        """
        if self.perceptron == None:
            messagebox.showerror("Error", "Please train the model first")
            return
        f1 = self.selected_feature(self.feature1.get())
        f2 = self.selected_feature(self.feature2.get())
        class1 = self.class1.get()
        class2 = self.class2.get()

        x, y, x_train, y_train, x_test, y_test = self.dataframe_to_xy(self.df, f1, f2, class1, class2)
        w1, w2, b = self.perceptron.get_weights()

        x1 = x_test[:,0]
        x2 = -(w1/w2) * x1 - (b/w2)

        pred = self.perceptron.prediction(x_test)

        cm = self.confusion_matrix(y_test, pred)

        print(f"Confusion Matrix: {cm['matrix']} \n Accuracy: {cm['accuracy']} Precision: {cm['precision']} recall: {cm['recall']} f1: {cm['f1']}")

        #plot the data
        plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color='blue', label=class1)
        plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color='orange', label=class2)

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f'{f1} vs {f2} for {class1} and {class2}')
        plt.legend(title="Classes")
        plt.plot(x1, x2, c='red')        
        plt.show()
    
    def test_adaline(self):
        """
        Test the adaline and calculate the confusion matrix.
        """
        if self.adaline == None:
            messagebox.showerror("Error", "Please train the model first")
            return
        f1 = self.selected_feature(self.feature1.get())
        f2 = self.selected_feature(self.feature2.get())
        class1 = self.class1.get()
        class2 = self.class2.get()

        x, y, x_train, y_train, x_test, y_test = self.dataframe_to_xy(self.df, f1, f2, class1, class2)
        w1, w2, b = self.adaline.get_weights()

        x1 = x_test[:,0]
        x2 = (0.5 - w1 * x1 - b) * (1/w2)
        pred = self.adaline.prediction(x_test)

        cm = self.confusion_matrix(y_test, pred)

        print(f"Confusion Matrix: {cm['matrix']} \n Accuracy: {cm['accuracy']} Precision: {cm['precision']} recall: {cm['recall']} f1: {cm['f1']}")

        #plot the data
        plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color='blue', label=class1)
        plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color='orange', label=class2)

        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f'{f1} vs {f2} for {class1} and {class2}')
        plt.legend(title="Classes")
        plt.plot(x1, x2, c='red')        
        plt.show()

    def confusion_matrix(self, y, pred):
        """
        Calculate the confusion matrix.

        Parameters
        ----------
        y : array-like
            The actual values.
        pred : array-like
            The predicted values.

        Returns
        -------
        dict
            The confusion matrix.
        """
        actual = np.array(y)
        predicted = np.array(pred)
        
        if self.radio_var.get() == "adaline":
            predicted = (predicted >= 0.5).astype(int)

        tp = np.sum((actual == 1) & (predicted == 1))
        tn = np.sum((actual == 0) & (predicted == 0))
        fp = np.sum((actual == 0) & (predicted == 1))
        fn = np.sum((actual == 1) & (predicted == 0))

        matrix = np.array([[tp, fp], [fn, tn]])

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'matrix': matrix,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn
        }
        return metrics

    def plot(self):
        """
        Plot the data.
        """
        f1 = self.selected_feature(self.feature1.get())
        f2 = self.selected_feature(self.feature2.get())
        class1 = self.class1.get()
        class2 = self.class2.get()

        data_loader.display_data(self.df, f1, f2, class1, class2)