import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

clusters_number = 3 # or 'auto'
questions = ['Вид источников для самостоятельного изучения', 'Способ организации учебного процесса', \
             'Численность учебной группы', 'Технология преподнесения материалов', 'Способ проверки знаний', \
             'Темп подачи материала', 'Форма обучения']

df = pd.read_excel('survey_data.xlsx', header=[0,1])
new_columns = ['No', 'ID', 'User', 'IP', 'Date', 'Course', 'Group']
question_num_set = set()
for column_tuple in list(df.columns):
    if 'Вопрос' in column_tuple[0]:
        question_num_set.add(column_tuple[0][-1])
        new_columns.append(f'{column_tuple[0][-1]}_{column_tuple[1]}')
df.set_axis(new_columns, axis=1, inplace=True)
df.drop(['No', 'ID', 'User', 'IP', 'Date', 'Course', 'Group'], axis=1, inplace=True)
question_number = len(question_num_set)

if clusters_number == 'auto':
    search_range = range(2, 21)
    report = {}
    for k in search_range:
        temp_dict = {}
        kmeans = KMeans(init='k-means++',
                        algorithm='auto',
                        n_clusters=k,
                        max_iter=1000,
                       random_state=1).fit(df)
        inertia = kmeans.inertia_
        temp_dict['Sum of squared error'] = inertia
        try:
            cluster = kmeans.predict(df)
            chs = calinski_harabasz_score(df, cluster)
            ss = silhouette_score(df, cluster)
            temp_dict['Calinski Harabasz Score'] = chs
            temp_dict['Silhouette Score'] = ss
            report[k] = temp_dict
        except:
            report[k] = temp_dict

    report_df = pd.DataFrame(report).T
    report_df.plot(figsize=(15, 10),
                   xticks=search_range,
                   grid=True,
                   title=f'Selecting optimal "K"',
                   subplots=True,
                   marker='o',
                   sharex=True)
    plt.tight_layout()
    chs = [-10, -10]
    ss = [-10, -10]
    for i in range(2, len(report)):
        chs.append(report[i]['Calinski Harabasz Score'])
        ss.append(report[i]['Silhouette Score'])
    chs = np.array(chs)
    ss = np.array(ss)
    clusters_number = (np.argmax(chs) + np.argmax(ss)) // 2

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=list(df.columns))
centroids_dict = centroids.to_dict(orient='records')

trajectories = {}
threshold = 0.2
for i in range(centroids.shape[0]):
    prev_question_index = 1
    trajectories[i+1] = {}
    for j in range(1, question_number+1):
        sub_dict = {}
        for key, value in centroids_dict[i].items():
            if f'{j}_' in key:
                sub_dict[key] = value
        best_names = [max(sub_dict, key=sub_dict.get)]
        best_val = max(sub_dict.values())
        for key, value in sub_dict.items():
            if (best_val - value) <= threshold and key not in best_names:
                best_names.append(key)
        trajectories[i+1][j] = best_names

for key, trajectory in trajectories.items():
    print(f'Траектория {key}:')
    for questiond_idx, answers in trajectory.items():
        answer = ' И/ИЛИ '.join([x[2:] for x in answers])
        print(f'- {questions[questiond_idx-1]}: {answer}')
#         print(f'- {answer}')
    print()