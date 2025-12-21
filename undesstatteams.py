import asyncio
import json
import nest_asyncio
import pandas as pd
import codecs

import aiohttp

from understat import Understat

shots_lst=[]
shots_sorted_lst=[]
shots_sorted=[]
columns2df=['minute', 'X', 'Y', 'player', 'shotType', 'result', 'h_team', 'a_team', 'player_assisted', 'xG']
shots_short=pd.DataFrame(columns=columns2df)
shots_clean_list=[]
own_goals_against=[]
own_goals_for=[]


#club="Bayer Leverkusen"
with open('club_name.txt', 'r') as f:
    club = f.read().strip()
    if not club:
        raise ValueError("club_name.txt is empty")
print(club)


async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        results = await understat.get_team_results(club, 2025)
        if results == "NoneType":
            raise ValueError(f"Failed to retrieve results for club '{club}'. Check if the club name is correct and the API is accessible.")
        clubmatches = json.dumps(results, indent=4, ensure_ascii=False)
        print(clubmatches)
        with codecs.open(f"{club}_matches.json", "w", "utf-8") as jsonfile:
            jsonfile.write(clubmatches)


if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    
matches_df=pd.read_json(f'{club}_matches.json')

#own goals computed separately from transfermarkt https://www.transfermarkt.com/bundesliga/eigentorstatistik/wettbewerb/L1/plus/?saison_id=2024
#own_goals=pd.read_json('own_goals_bl.json')

#for key, value in enumerate(own_goals['own_goals']):
#    own_goals_club.append(value['Club'])
#    own_goals_for.append(value['Tore_durch_Eigentor'])

#for i in range(0,len(own_goals_club)):
#    if own_goals_club[i]==club:
#        own_goals_total=own_goals_for[i]


matches_ids=matches_df['id']

async def main():
    async with aiohttp.ClientSession() as session:
        for i, value in enumerate(matches_ids):
            understat2 = Understat(session)
            shots = await understat2.get_match_shots(value)
            shots_lst.append(shots)
            

if __name__ == "__main__":
    nest_asyncio.apply()
    loop2 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop2)
    loop.run_until_complete(main())
    

def sort_home_away():
    for i, value in enumerate(matches_df['side']):
        shots_sorted=shots_lst[i][value]
        shots_sorted_lst.append(shots_sorted)
    return shots_sorted_lst


def clean_shots_df():
    for match in shots_sorted_lst:
        for i, shots in enumerate(match):
            player=shots['player']
            result=shots['result']
            minute=shots['minute']
            X=shots['X']
            Y=shots['Y']
            shotType=shots['shotType']
            h_team=shots['h_team']
            a_team=shots['a_team']
            player_assisted=shots['player_assisted']
            xG=shots['xG']
            clean_shots=shots_short.loc[i, columns2df]=[minute, X, Y, player, shotType, result, h_team, a_team, player_assisted, xG]
            shots_clean_list.append(clean_shots)
    #df_shots_clean=pd.DataFrame(shots_clean_list)
    #return df_shots_clean

for match in shots_lst:
    for home in match['h']:
        if home['result']=='OwnGoal':
            if(home['h_team'])!=club:
                own_goals_for.append(home)
            else:
                own_goals_against.append(home)
    for away in match['a']:
        if away['result']=='OwnGoal':
            if(away['a_team'])!=club:
                own_goals_for.append(away)
            else:
                own_goals_against.append(away)
                

sort_home_away()
clean_shots_df()
df_own_goals_for=pd.DataFrame(own_goals_for, columns=columns2df)
df_shots_clean=pd.DataFrame(shots_clean_list, columns=columns2df)

shots_with_own_goal=pd.concat([df_own_goals_for, df_shots_clean])
df=pd.DataFrame(shots_with_own_goal, columns=columns2df)
df.to_csv(f'{club}_seasons_shots.csv', index=False)         