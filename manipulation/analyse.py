from __future__ import annotations

# trunk-ignore(flake8/F401)
import io

# trunk-ignore(flake8/F401)
import json

# trunk-ignore(flake8/F401)
import subprocess

# trunk-ignore(flake8/F401)
from functools import reduce
from pprint import pprint

# trunk-ignore(flake8/F401)
import matplotlib.pyplot as plt

# trunk-ignore(flake8/F401)
import numpy as np

# trunk-ignore(flake8/F401)
import pandas as pd
import seaborn as sns

# trunk-ignore(flake8/F401)
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud

sns.set_theme(style="darkgrid")

from typing import TYPE_CHECKING

from . import utils

if TYPE_CHECKING:
    from pathlib import Path


def analyse(training_folder: Path, testing_folder: Path, analyse_folder: Path):

    train_df, test_df = utils.get_data(training_folder, testing_folder)

    pprint(train_df[["content", "prediction"]].describe())

    spams = train_df[train_df.prediction == "spam"]
    hams = train_df[train_df.prediction == "ham"]

    print("Counting spam vs ham in training dataset...")
    count_plot_path = analyse_folder.joinpath("count_plot.png")
    count_plot = sns.countplot(x="prediction", data=train_df)
    count_plot.set(xlabel="class")
    utils.save_plot(count_plot_path)
    print(f" plot saved in {count_plot_path}")

    # Instantiate a new wordcloud.
    wordcloud = WordCloud(
        random_state=42, normalize_plurals=False, width=600, height=300
    )

    print("Generating wordcloud from training dataset's spams...")
    spams_wordcloud_path = analyse_folder.joinpath("spams_wordcloud.png")
    text = " ".join(content for content in spams.content)
    wordcloud.generate(text)
    wordcloud.to_file(spams_wordcloud_path)
    print(f" plot saved in {spams_wordcloud_path}")

    print("Generating wordcloud from training dataset's hams...")
    hams_wordcloud_path = analyse_folder.joinpath("hams_wordcloud.png")
    text = " ".join(content for content in hams.content)
    wordcloud.generate(text)
    wordcloud.to_file(hams_wordcloud_path)
    print(f" plot saved in {hams_wordcloud_path}")

    print("Generating content length plot on training dataset...")
    content_length_path = analyse_folder.joinpath("content_length.png")
    sns.histplot(
        data=train_df,
        hue="prediction",
        hue_order=["ham", "spam"],
        x="content_length",
        log_scale=True,
        element="step",
        fill=False,
        cumulative=False,
        stat="density",
        common_norm=False,
    )
    utils.save_plot(content_length_path)
    print(f" plot saved in {content_length_path}")

    """
    y_train = train_df.prediction
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    for json_file in analyse_folder.glob("*.json"):
        with json_file.open() as fp:
            gs_results = json.load(fp)
        
        params = gs_results['params']
        scores = gs_results['mean_test_score']
        best_params = params[np.argmin(gs_results['rank_test_score'])]
        best_params['clf__max_depth'] = -1 if best_params['clf__max_depth'] is None else best_params['clf__max_depth']
        
        plot_df = pd.concat((
            pd.DataFrame.from_records(params).fillna(-1),
            pd.DataFrame(scores, columns=['score'])
        ), axis='columns')
        
        for plot_param, param_value in best_params.items():
            if plot_param == 'clf__max_depth':
                continue
            
            of_interest_df = plot_df.loc[reduce(lambda a, b: a & b, (plot_df[param_name] == param_value for param_name, param_value in best_params.items() if param_name != plot_param))]
            of_interest_df.reset_index(drop=True, inplace=True)
            
            for index, rows in of_interest_df.iterrows():
                parameters = rows.astype(int).to_dict()
                del parameters['score']
                parameters['clf__max_depth'] = None if parameters['clf__max_depth'] == -1 else parameters['clf__max_depth']
                pipeline = utils.generate_pipeline(clf)
                pipeline.set_params(**parameters)
                
                pred_df, _, _ = utils.fit_and_predict(pipeline, train_df, y_train, test_df)
                
                formatted = pred_df.applymap(lambda x: int(x != "spam"))
                tmp_path = analyse_folder.joinpath("tmp.csv")
                formatted.to_csv(tmp_path, index_label="id")
                subprocess.run(["kaggle", "competitions", "submit", "-c", "adcg-ss14-challenge-02-spam-mails-detection", "-f", str(tmp_path.resolve()), "-m", json.dumps(parameters)])
            
            kaggle_process = subprocess.run(["kaggle", "competitions", "submissions", "-c", "adcg-ss14-challenge-02-spam-mails-detection", "-v"], capture_output=True, text=True)
            kaggle_results_csv = io.StringIO(kaggle_process.stdout)
            kaggle_results = pd.read_csv(kaggle_results_csv, nrows=len(of_interest_df), usecols=['publicScore'])[::-1]
            kaggle_results.reset_index(drop=True, inplace=True)
            of_interest_df = pd.concat((
                of_interest_df,
                kaggle_results
            ), axis='columns')
            of_interest_df.rename(columns={'score': 'Local F1 score', 'publicScore': 'Kaggle accuracy score'}, inplace=True)
            
            param_plot_path = analyse_folder.joinpath(f'{plot_param}_plot.png')
            f1_plot = sns.lineplot(x=plot_param, y="Local F1 score", data=of_interest_df, color='blue')
            kaggle_plot = sns.lineplot(x=plot_param, y="Kaggle accuracy score", data=of_interest_df, color='green')
            sns.scatterplot(data={param_value: of_interest_df['Local F1 score'].max()}, legend=False, zorder=10, color="red")
            f1_plot.set(xlabel=plot_param.replace('_', ' '), ylabel="scores")
            plt.legend(loc='best', labels=['Local F1 score', 'Kaggle accuracy score'])
            utils.save_plot(param_plot_path)
    """
