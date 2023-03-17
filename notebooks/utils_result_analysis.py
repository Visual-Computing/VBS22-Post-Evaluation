import numpy as np
import itertools
import pandas as pd

##Functions for data processing

##### TAKS ######
def get_task_from_taskname(tasks_df, taskname):
        """
        :param tasks_df:  dataframe containing all the tasks.
        tasks_df.columns=['name', 'started', 'ended', 'duration', 'position', 'uid', 'task_type',
             'correct_video', 'fps', 'correct_shot', 'target_start_ms', 'target_end_ms', 'submissions', 'hints']
        :param taskname: name of the task, e.g., "vbs2001"
        :return: dictionary with the info on the task corresponding to "taskname"
        """
        return tasks_df[tasks_df['name'] == taskname].iloc[0].to_dict()

##### SUBMISSIONS ######
# def get_submissions_count(sub_df):
#         """
#         :param sub_df: dataframe wil all the submissions (columns:['taskName', 'team', 'teamFamily', 'user', 'task_start', 'task_end',
#        'timestamp', 'sessionID', 'status'])
#         :return: a dataframe with stats on submissions for each team, user and task. (columns: ['task', 'team', 'teamFamily', 'user', 'nCorrectSubmissions',
#        'nWrongSubmissions'])
#         """
#         wrong_sub=sub_df[sub_df['status']=='WRONG']
#         wrong_sub=wrong_sub.groupby(["taskName","team","teamFamily","user"])[['status']].count()
#         wrong_sub.columns = ["nWrongSubmissions"]
#         correct_sub=sub_df[sub_df['status']=='CORRECT']
#         correct_sub=correct_sub.groupby(["taskName","team","teamFamily","user"])[['status']].count()
#         correct_sub.columns = ["nCorrectSubmissions"]
#
#         correct_sub=correct_sub.reset_index()
#         wrong_sub=wrong_sub.reset_index()
#         submissions=correct_sub.merge(wrong_sub, on=["taskName","team","teamFamily","user"], how="outer").fillna(0)
#         submissions=submissions.rename(columns={"taskName": "task"})
#         return  submissions.reset_index().drop(columns=['index'])


def dres_KIS_score(index_firstCorrect,time_correct_submission, tDur):
        """
        :param index_firstCorrect: index of first CORRECT submission. It is -1 if there are no corret submission,  ortherwise it is equal to the number of WRONG submissions before the CORRECT one
        :param time_correct_submission:
        :param tDur: actual duration of task (in case it was extended during competition)
        :return: score assigned to the team
        """
        #todo: to be checked I used a formula found at https://github.com/dres-dev/DRES/blob/37bfa448852a090c564b7519b8c08292f71ede36/backend/src/main/kotlin/dev/dres/run/score/scorer/KisTaskScorer.kt
        maxPointsPerTask = 100.0
        maxPointsAtTaskEnd = 50.0
        penaltyPerWrongSubmission = 10.0

        score=0.0
        if index_firstCorrect>-1:
                timeFraction =1.0-time_correct_submission/tDur
                score=  max(0.0,maxPointsAtTaskEnd + ((maxPointsPerTask - maxPointsAtTaskEnd) * timeFraction) - (index_firstCorrect * penaltyPerWrongSubmission) )
        return score

def compute_team_scores(task_list, teams_list,sub_df):
        """
        create a dataframe that specifies for each task and team what is the achieved DRES score
        :param task_list: list of task names
        :param task_list: list of team names
        :param sub_df: submissions dataframe (one row for each submission)(columns=['taskName', 'team', 'teamFamily', 'user', 'task_start', 'task_end',
       'timestamp', 'sessionID', 'status'])
        :return: a dataframe that specifies for each task and team what is the achieved DRES score
        """
        scores=[]

        for task,team in itertools.product(task_list, teams_list):
                sub_team_task=sub_df.loc[(sub_df['team'].str.startswith(team)) & (sub_df['taskName']==task) ]
                sub_team_task=sub_team_task.sort_values(by=['timestamp']).reset_index()
                sub_team_task.drop(columns=['index'], inplace=True)

                score_team_task=0.0
                if len(sub_team_task.index)>0:
                        correct_sub=sub_team_task [sub_team_task['status']=='CORRECT'].reset_index()
                        if(len(correct_sub.index)>0):
                                firstCorrect=correct_sub.iloc[0]
                                timestamp=firstCorrect['timestamp']
                                time_correct_submission=timestamp-firstCorrect['task_start'] #milliseconds
                                tDur=firstCorrect['task_end']-firstCorrect['task_start']
                                index_firstCorrect=firstCorrect['index']
                                score_team_task=dres_KIS_score(index_firstCorrect,time_correct_submission, tDur)

                scores.append({'team': team, 'task':task, 'score': score_team_task })
        return pd.DataFrame(scores)


######### PROCESSING TEAM LOGS #################
def get_team_values_df(df, task_df, max_rank=10000):
        """
        :param df:
        :param task_df:
        :param max_rank:
        :return: dataframe containing results for all teams and tasks
        """
        #remove ranks bigger than max_rank
        replace_large_ranks = lambda x: np.inf if x > max_rank  else x
        df['rank_video'] = df['rank_video'].apply(replace_large_ranks)
        df['rank_shot_margin_0'] = df['rank_shot_margin_0'].apply(replace_large_ranks)
        df['rank_shot_margin_5'] = df['rank_shot_margin_5'].apply(replace_large_ranks)

        # for each (team, user, task), find the minimum ranks and the timestamps
        df=df.sort_values('timestamp')
        best_video_df = df.loc[df.groupby(['teamFamily','team', 'user', 'task'])['rank_video'].idxmin()]
        best_shot_df = df.loc[df.groupby(['teamFamily','team', 'user', 'task'])['rank_shot_margin_0'].idxmin()]
        best_shot_df_5secs = df.loc[df.groupby(['teamFamily','team', 'user','task'])['rank_shot_margin_5'].idxmin()]
        # find also the time of first and last appearance of a result in the ranked list
        df_valid_rankshot = df[~df['rank_shot_margin_0'].isin([np.inf, -np.inf])]
        df_valid_rankvideo = df[~df['rank_video'].isin([np.inf, -np.inf])]
        first_appearance_time = df_valid_rankshot.loc[df_valid_rankshot.groupby(['teamFamily','team', 'user', 'task'])['timestamp'].idxmin()]
        first_appearance_time_video = df_valid_rankvideo.loc[df_valid_rankvideo.groupby(['teamFamily','team', 'user', 'task'])['timestamp'].idxmin()]
        last_appearance_time = df_valid_rankshot.loc[df_valid_rankshot.groupby(['teamFamily','team', 'user', 'task'])['timestamp'].idxmax()]

        best_video_df = best_video_df.filter(['teamFamily','team', 'user', 'task', 'rank_video', 'timestamp', 'correct_submission_time_ms']).rename(
                columns={'timestamp': 'timestamp_best_video'})
        best_shot_df = best_shot_df.filter(['teamFamily','team', 'user','task', 'rank_shot_margin_0', 'timestamp']).rename(
                columns={'timestamp': 'timestamp_best_shot'})
        best_shot_df_5secs = best_shot_df_5secs.filter(
                ['teamFamily','team', 'user', 'task', 'rank_shot_margin_5', 'timestamp']).rename(
                columns={'timestamp': 'timestamp_best_shot_5secs'})
        first_appearance_time = first_appearance_time.filter(['teamFamily','team', 'user', 'task', 'timestamp', 'rank_shot_margin_0']).rename(
                columns={'timestamp': 'timestamp_first_appearance', 'rank_shot_margin_0': 'rank_shot_first_appearance'})
        first_appearance_time_video = first_appearance_time_video.filter(['teamFamily','team', 'user', 'task', 'timestamp', 'rank_video']).rename(
                columns={'timestamp': 'timestamp_first_appearance_video', 'rank_video': 'rank_video_first_appearance'})
        last_appearance_time = last_appearance_time.filter(['teamFamily','team', 'user', 'task', 'timestamp', 'rank_shot_margin_0']).rename(
                columns={'timestamp': 'timestamp_last_appearance', 'rank_shot_margin_0': 'rank_shot_last_appearance'})

        #setting best timestamp to np.inf if there is not a best video/shot
        best_video_df.loc[df['rank_video'].isin([np.inf, -np.inf]), 'timestamp_best_video'] = -1
        best_shot_df.loc[df['rank_shot_margin_0'].isin([np.inf, -np.inf]), 'timestamp_best_shot']=-1
        best_shot_df_5secs.loc[df['rank_shot_margin_5'].isin([np.inf, -np.inf]), 'timestamp_best_shot_5secs'] = -1

        df = best_video_df.merge(best_shot_df, on=['teamFamily','team', 'user', 'task'])
        df = df.merge(best_shot_df_5secs, on=['teamFamily','team', 'user','task'])
        df = df.merge(first_appearance_time, on=['teamFamily','team', 'user','task'], how="outer")
        df = df.merge(last_appearance_time, on=['teamFamily','team', 'user','task'], how="outer")
        df = df.merge(first_appearance_time_video, on=['teamFamily','team', 'user','task'], how="outer")

        # convert timestamps in actual seconds from the start of the task
        df['task_start'] = df['task'].apply(lambda x: get_task_from_taskname(task_df,x)['started'])
        df['time_best_video'] = (df['timestamp_best_video'] - df['task_start'])
        df['time_best_shot'] = (df['timestamp_best_shot'] - df['task_start'])
        df['time_first_appearance'] = (df['timestamp_first_appearance'] - df['task_start'])
        df['time_first_appearance_video'] = (df['timestamp_first_appearance_video'] - df['task_start'])
        df['time_last_appearance'] = (df['timestamp_last_appearance'] - df['task_start'])
        df['time_best_shot_margin5'] = (df['timestamp_best_shot_5secs'] - df['task_start'])
        #df['time_correct_submission'] = df.apply(lambda x: runreader.get_csts()[x['teamFamily','team']][x['task']] -
        #                                                   runreader.tasks.get_task_from_taskname(x['task'])[
        #                                                           'started'], axis=1)
        fix_time_fun=lambda x: x / 1000 if x > 0 else np.inf
        df['time_best_video'] = df['time_best_video'].astype(float).apply(fix_time_fun)
        df['time_best_shot'] = df['time_best_shot'].astype(float).apply(fix_time_fun)
        df['time_best_shot_margin5'] = df['time_best_shot_margin5'].astype(float).apply(fix_time_fun)
        df['time_correct_submission'] = df['correct_submission_time_ms'].astype(float).apply(fix_time_fun)
        df['time_first_appearance'] = df['time_first_appearance'].astype(float).apply(fix_time_fun)
        df['time_first_appearance_video'] = df['time_first_appearance_video'].astype(float).apply(fix_time_fun)
        df['time_last_appearance'] = df['time_last_appearance'].astype(float).apply(fix_time_fun)

        df = df.round(decimals=0)

        df = df.filter(['teamFamily','team', 'user', 'task', 'task_start', 'time_correct_submission', 'time_best_video', 'time_best_shot',
                        'time_best_shot_margin5', 'rank_video', 'rank_shot_margin_0', 'rank_shot_margin_5',
                        'rank_shot_margin_10', 'time_first_appearance', 'rank_shot_first_appearance', 'time_last_appearance', 'rank_shot_last_appearance',
                        'time_first_appearance_video', 'rank_video_first_appearance'])

        df.replace([np.inf, -np.inf, np.nan], -1, inplace=True)

        return df

######### DF for tables and plots #################
def TimeRecallTable(df,teams):



        #maniant only useful columns
        df=df[['team', 'user', 'task',
        'time_correct_submission', 'time_best_video', 'time_best_shot',
        'time_best_shot_margin5', 'rank_video', 'rank_shot_margin_0',
        'rank_shot_margin_5']]
        # drop unuseful columns from df that has the structure of df_results
        # df = df.drop(
        #     ['time_first_appearance', 'rank_shot_first_appearance', 'time_last_appearance', 'rank_shot_last_appearance',
        #      'time_first_appearance_video', 'rank_video_first_appearance'], axis=1)
        #df.drop(columns='task_start', inplace=True)

        df = df.fillna(-1)
        col = [c for c in df.columns.values.tolist() if c != 'team' and c != 'task' and c != 'user' and c != 'teamFamily' ]
        df[col] = df[col].astype('int32')
        df[col] = df[col].applymap(lambda x: -1 if x < 0 else x)
        df = df.astype('str')
        df.replace(['-1'], '-', inplace=True)


        # aggregate
        agg_dic = {c: (lambda x: ' / '.join(x)) for c in col}
        agg_dic['time_correct_submission'] = "min"
        df = df.groupby(['team', 'task'])[col].agg(agg_dic).reset_index()
        df.replace('- / -', '-', regex=True, inplace=True)
        add_second = lambda x: x if x == '-' else x + 's'
        df['time_correct_submission'] = df['time_correct_submission'].apply(add_second)
        df['time_best_shot'] = df['time_best_shot'].apply(add_second)
        df['time_best_video'] = df['time_best_video'].apply(add_second)
        df = df.melt(var_name="metric", id_vars=["team", "task"], value_name="value")
        df['unit'] = df['metric'].apply(lambda x: 'rank' if x.startswith('rank_') else 'time')
        replace_dic = {
            'rank_shot_margin_0': 'correct frame',
            'time_best_shot': 'correct frame',
            'rank_shot_margin_5': 'frame in GT+2x5s',
            'time_best_shot_margin5': 'frame in GT+2x5s',
            'rank_video': 'correct video',
            'time_best_video': 'correct video',
            'time_correct_submission': 'correct submission'
        }
        df['metric'] = df['metric'].map(replace_dic)
        df = df.pivot(index=['team', 'metric', 'unit'], columns="task", values="value")
        df = df.fillna('!')

        # sorting index desired order
        # level_0 = teams  # order in the conf file
        # level_1 = ['correct frame', 'frame in GT+2x5s', 'correct video','correct submission']
        # level_2 = ['rank', 'time']
        # df = df.reindex(pd.MultiIndex.from_product([level_0, level_1, level_2]))
        # df.dropna(axis=0, inplace=True)  # 'correct submission'/rank shluld not be in the index
        # print(df)
        # print(f"Saving: {output_dir}/time_recall_table_withMargin5_vbse2022.csv")
        # df.to_csv(f"{output_dir}/time_recall_table_withMargin5_vbse2022.csv")
        # sorting index desired order
        level_0 = teams  # order in the conf file
        level_1 = ['correct frame', 'correct video','correct submission']
        level_2 = ['rank', 'time']
        df = df.reindex(pd.MultiIndex.from_product([level_0, level_1, level_2]))
        df.dropna(axis=0, inplace=True)  # 'correct submission'/rank shluld not be in the index
        return df

# def BestShotRankBoxplot(df,teams, max_records=10000, split_user=True):
#         view = df[df["rank_shot_margin_0"] != -1].groupby(['team', 'user']).agg('count')[
#                 'rank_shot_margin_0']
