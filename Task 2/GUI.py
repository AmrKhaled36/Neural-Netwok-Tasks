import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data_loader
from neuron import ActivationFunction, Neuron
from edge import Edge

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
        self.root.title("Task 2 NN")
        self.df = df
        self.layers = []
        
        for i in range(6):
            self.root.grid_rowconfigure(i, weight=1)
        for i in range(6):
            self.root.grid_columnconfigure(i, weight=1)

        #First row

        self.hidden_layers_label = tk.Label(self.root, text="# Hidden self.layers:")
        self.hidden_layers_label.grid(row=0, column=0, padx=10, pady=10)

        self.hidden_layers_entry = tk.Entry(self.root)
        self.hidden_layers_entry.grid(row=1, column=0, padx=10, pady=10)

        self.neurons_label = tk.Label(self.root, text="Enter neurons per layer (comma separated):")
        self.neurons_label.grid(row=0, column=1, padx=10, pady=10)

        self.neurons_entry = tk.Entry(self.root)
        self.neurons_entry.grid(row=1, column=1, padx=10, pady=10)

        self.learning_rate_label = tk.Label(self.root, text="Learning rate:")
        self.learning_rate_label.grid(row=0, column=2, padx=10, pady=10)

        self.learning_rate_entry = tk.Entry(self.root)
        self.learning_rate_entry.grid(row=1, column=2, padx=10, pady=10)

        self.epochs_label = tk.Label(self.root, text="Epochs:")
        self.epochs_label.grid(row=0, column=3, padx=10, pady=10)

        self.epochs_entry = tk.Entry(self.root)
        self.epochs_entry.grid(row=1, column=3, padx=10, pady=10)

        self.bias_var = tk.BooleanVar()
        self.bias_check = tk.Checkbutton(self.root, text="Bias", variable=self.bias_var)
        self.bias_check.grid(row=2, column=0, padx=10, pady=10)

        self.radio_var = tk.StringVar()
        self.radio_var.set("Sigmoid")

        self.radio_Sigm = tk.Radiobutton(self.root, text="Sigmoid", variable=self.radio_var, value="Sigmoid")
        self.radio_Sigm.grid(row=2, column=1, padx=10, pady=10)

        self.radio_Tanh = tk.Radiobutton(self.root, text="Tanh", variable=self.radio_var, value="Tanh")
        self.radio_Tanh.grid(row=3, column=1, padx=10, pady=10)

        self.train_button = tk.Button(self.root, text="Train", command=self.train, width=10)
        self.train_button.grid(row=3, column=2, padx=10, pady=10)

        self.predict1_entry = tk.Entry(self.root)
        self.predict1_entry.grid(row=5, column=0, padx=10, pady=10)

        self.predict1_label = tk.Label(self.root, text="gender")
        self.predict1_label.grid(row=4, column=0, padx=10, pady=10)

        self.predict2_entry = tk.Entry(self.root)
        self.predict2_entry.grid(row=5, column=1, padx=10, pady=10)

        self.predict2_label = tk.Label(self.root, text="Body Mass")
        self.predict2_label.grid(row=4, column=1, padx=10, pady=10)

        self.predict3_entry = tk.Entry(self.root)
        self.predict3_entry.grid(row=5, column=2, padx=10, pady=10)

        self.predict3_label = tk.Label(self.root, text="Beak Length")
        self.predict3_label.grid(row=4, column=2, padx=10, pady=10)

        self.predict4_entry = tk.Entry(self.root)
        self.predict4_entry.grid(row=5, column=3, padx=10, pady=10)

        self.predict4_label = tk.Label(self.root, text="Beak Depth")
        self.predict4_label.grid(row=4, column=3, padx=10, pady=10)

        self.predict5_entry = tk.Entry(self.root)
        self.predict5_entry.grid(row=5, column=4, padx=10, pady=10)

        self.predict5_label = tk.Label(self.root, text="Fin Length")
        self.predict5_label.grid(row=4, column=4, padx=10, pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict, width=10)
        self.predict_button.grid(row=5, column=5, padx=10, pady=10)

    def predict(self):
        gender = self.predict1_entry.get().lower()
        x1 = 0 if gender == "female" else 1
        x2 = float(self.predict2_entry.get())
        x3 = float(self.predict3_entry.get())
        x4 = float(self.predict4_entry.get())
        x5 = float(self.predict5_entry.get())
        
        x1,x2,x3,x4,x5 = data_loader.normalize_minmax(self.df, [x1,x2,x3,x4,x5])

        i = np.array([x1,x2,x3,x4,x5])
        predicted_values = []

        for indx,k in enumerate(self.layers[0]):
            k.output = i[indx]
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.calc_output()
        l = []
        for neuron in self.layers[-1]:
            l.append(neuron.output)
        max_value = max(l)
        max_index = l.index(max_value)
        if max_index == 0:
            messagebox.showinfo("Prediction",f"Predicted Class A")
        elif max_index == 1:
            messagebox.showinfo("Prediction",f"Predicted Class B")
        elif max_index == 2:
            messagebox.showinfo("Prediction",f"Predicted Class C")


    def dataframe_to_xy(self, df):
        """
        Return the x, y, x_train, y_train, x_test, y_test.
        """
        x, y, x_train, y_train, x_test, y_test = data_loader.load_x_y(df)
        return x, y, x_train, y_train, x_test, y_test
    
    def train(self):
        """
        Train the neural network.
        """
        lr = float(self.learning_rate_entry.get())
        epochs = int(self.epochs_entry.get())
        bias = self.bias_var.get()
        if self.neurons_entry.get() != "":
            neurons_per_layer = [int(i) for i in self.neurons_entry.get().split(",")]
        else:
            neurons_per_layer = [10]
        
        if self.radio_var == "Sigmoid":
            activation_function = ActivationFunction.sigmoid
            self.train_NN(lr, epochs, bias, activation_function, self.hidden_layers_entry.get(), neurons_per_layer)
        else:
            activation_function = ActivationFunction.tanh
            self.train_NN(lr, epochs, bias, activation_function, self.hidden_layers_entry.get(), neurons_per_layer)

    def calculate_metrics(self, confusion_matrix):
        # Accuracy
        total_correct = np.trace(confusion_matrix)  # Sum of diagonal elements
        total_samples = np.sum(confusion_matrix)
        accuracy = total_correct / total_samples

        # Precision, Recall, and F1-Score for each class
        num_classes = confusion_matrix.shape[0]
        precision = []
        recall = []
        f1_score = []
        
        for i in range(num_classes):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score_i = 2 * (precision_i * recall_i) / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0
            
            precision.append(precision_i)
            recall.append(recall_i)
            f1_score.append(f1_score_i)
        
        # Print metrics
        print(confusion_matrix)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1_score)

    def train_NN(self, lr, epochs, bias = 0, activation_function = ActivationFunction.sigmoid, hidden_layers = 1, neurons_per_layer = [10]):
        """
        Train the neural network.
        """
        x, y, x_train, y_train, x_test, y_test = self.dataframe_to_xy(self.df)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        neurons_per_layer.append(3)
        self.layers = []
        Edge.lr = lr
        
        l = []
        #construct the layers
        for j in x_train[0]:
            feat = Neuron(None, 0, None)
            feat.output = j
            l.append(feat)
        self.layers.append(l)

        for k in neurons_per_layer:
            layer = []
            for j in range(k):
                layer.append(Neuron(self.layers[-1],bias,activation_function))
            self.layers.append(layer)
        

        for i in range(epochs):

            
            for j in range(len(x_train)):
                
                self.layers[-1][0].target = 1 if y_train[j] == 0 else 0
                self.layers[-1][1].target = 1 if y_train[j] == 1 else 0
                self.layers[-1][2].target = 1 if y_train[j] == 2 else 0
                #Forward propagation
                for idx,k in enumerate(self.layers[0]):
                    self.layers[0][idx].output = x_train[j][idx]
                for layer in self.layers[1:]:
                    for neuron in layer:
                        neuron.calc_output()
                
                ##Backpropagation

                for layer in reversed(self.layers[1:]):
                    for neuron in layer:
                        neuron.calc_error_signal()

                #Update

                for layer in self.layers[1:]:
                    for neuron in layer:
                        neuron.update_weight()


        predicted_train_values = []
        for idx,i in enumerate(x_train):

            for indx,k in enumerate(self.layers[0]):
                k.output = i[indx]
            for layer in self.layers[1:]:
                for neuron in layer:
                    neuron.calc_output()
            l = []
            for neuron in self.layers[-1]:
                l.append(neuron.output)
            max_value = max(l)
            max_index = l.index(max_value)
            predicted_train_values.append(max_index)
        
        actual = np.array(y_train)
        
        predicted = np.array(predicted_train_values)
        print(actual.shape, predicted.shape)

        cm = np.zeros((3, 3), dtype=int)
        for true, pred in zip(actual, predicted):
            cm[true][pred] += 1

        # Calculate metrics
        print("#############################Train Metrics####################################")
        self.calculate_metrics(cm)



        predicted_values = []

        for idx,i in enumerate(x_test):

            for indx,k in enumerate(self.layers[0]):
                k.output = i[indx]
            for layer in self.layers[1:]:
                for neuron in layer:
                    neuron.calc_output()
            l = []
            for neuron in self.layers[-1]:
                l.append(neuron.output)
            max_value = max(l)
            max_index = l.index(max_value)
            predicted_values.append(max_index)


        #Confusion matrix
        
        actual = np.array(y_test)
        predicted = np.array(predicted_values)

        cm = np.zeros((3, 3), dtype=int)

        for true, pred in zip(actual, predicted):
            cm[true][pred] += 1

        # Calculate metrics
        print("#############################Test Metrics####################################")
        self.calculate_metrics(cm)

                
            






    