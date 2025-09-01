std_msgs/Header header
string intent   # "GAME_START"|"CARD_CHECK"|"PLAY_START"|"CALL"|"RAISE"|"FOLD"|"SET_RESULT"
int32  amount   # 0,1,2,3
string param    # SET_RESULT 전용 ("ROBOT_WIN"|"PLAYER_WIN"|"DRAW" or "")
