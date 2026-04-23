from data_loader import load_data
from preprocessing import preprocess
from experiment import run_experiment

def main():
    df = load_data("B.csv")
    df = preprocess(df)
    run_experiment(df)

if __name__ == "__main__":
    main()