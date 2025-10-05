from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def load_dataset():
    """
    Function reads the dataset 
    Returns:
        A loaded data frame
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    return df

def show_plots(df, x_name, y_name, hue_name):
    """
    Function illustrates the  data using: violin plot, box plot
    Parameters: 
        df - the data frame for which the plots are displayed
        x_name - variable to be plotted on the x-axis; categorical data
        y_name - variable to be plotted on the y-axis; quantitative data
        hue_name - categorical variable baesd on which the plot elements are coloured
    """
    fig, axes = plt.subplots(1, 2) 

    sns.violinplot(data=df, x=x_name, y=y_name, hue=hue_name, ax=axes[0])
    axes[0].set_title('Violin Plot')

    sns.boxplot(data=df, x=x_name, y=y_name, hue=hue_name, ax=axes[1])
    axes[1].set_title('Box Plot')

    plt.tight_layout()
    plt.show()

def main(dataset_name):
    """
    Main function of the project
    Parameters:
        dataset_name - name of the used dataset
    """
    if dataset_name == "iris": df = load_dataset()
    else: raise ValueError("Неизвестный датасет")

    show_plots(df, "species", "sepal length (cm)", "species")

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("Введите данные в виде: python main.py <имя_датасета>")
        
        dataset_name = sys.argv[1]
    
        main(dataset_name)
        sys.exit(0)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        sys.exit(1)
        
