# Copyright 2025 Han Junsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # 라이브러리 --------------------------------------------------->
import sys
import io
import csv
import json
import difflib
import shlex
import re
import random
import time
import math

#기능 함수 ----------------------------------------------------->
# sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 유사도 계산 
def is_correct(user_input, correct_answer):

    # 한글 자모 분리
    def decompose_hangul(text):
        result = []

        for char in text:
            if '가' <= char <= '힣':  # 완성형 한글
                char_code = ord(char) - 0xAC00
                cho = char_code // 588
                jung = (char_code % 588) // 28
                jong = char_code % 28

                result.append(chr(0x1100 + cho))  # 초성
                result.append(chr(0x1161 + jung)) # 중성

                if jong != 0:
                    result.append(chr(0x11A7 + jong)) # 종성

            else:
                result.append(char)

        return ''.join(result)
    

    user = decompose_hangul(user_input.strip().replace(" ", "")) #개행문자 제거, 띄어쓰기 제거
    correct = decompose_hangul(correct_answer.strip().replace(" ", ""))
    ratio = difflib.SequenceMatcher(None, user, correct).ratio() #유사도 측정

    len_correct = len(correct)
    if len_correct > 15: #단어가 길어 판정이 과하게 후해지는걸 방지
        len_correct = 15

    threshold = 0.989 ** len_correct #짧은 단어일수록 빡빡하게 판정, 길수록 유하게

    return ratio > threshold #판정치를 넘으면 정답 인정

# 입력 명령어 처리
def parse_command(user_input):
    # /명령어 추출
    tokens = shlex.split(user_input)
    command = tokens[0]
    
    # 옵션 딕셔너리 변환
    args = {}
    key = None
    for token in tokens[1:]:
        if token.startswith('-'):
            key = token.lstrip('-')
            args[key] = ""

        else:
            if args[key]:
                args[key] += " " + token

            else:
                args[key] = token
    
    return command, args
    # /addw
    # {'m': '사과', 'w': 'apple', 'e': 'I have a apple.'}

# wordlist.csv 파일에서 값 가져오기
def get_wordlist():
    with open('user_info/wordlist.csv', 'r', newline = '', encoding = 'utf-8') as file:
        reader = csv.DictReader(file)

        return list(reader)
        # [{'word': 'apple', 'mean': '사과', 'recent_results': '00000', 'last_quiz_timestamp': '0', 'is_selected': '1', 'example sentence': 'I have a apple.', 'predicted_accuracy': '0.0'}, ...]

# DEEPlearing accuracy prediction
def DEEP_AP(word_data):
    # 입력 데이터 파싱
    resent_results = list(map(int, list(word_data['recent_results'].zfill(5))))
    date_hours = (int(time.time()) - int(word_data['last_quiz_timestamp'])) / 3600  # 시간 단위
    streak = 0
    fail_streak = 0
    max_streak = 0
    max_fail_streak = 0
    past_eval = float(word_data['predicted_accuracy'])
    test_count = int(word_data.get('test', 0))

    # streak / fail_streak 계산
    for r in resent_results:
        if r == 1:
            streak += 1
            fail_streak = 0
        else:
            streak = 0
            fail_streak += 1
        max_streak = max(max_streak, streak)
        max_fail_streak = max(max_fail_streak, fail_streak)
    
    # --------- 모델 정보 불러오기 ----------
    with open('models\main_model.json', 'r') as file:
        data = json.load(file)

    # --------- 스케일링 Min/Max 값 ---------
    scaler_min = [
                data['model']['minmax']['min_resent_results'][0],
                data['model']['minmax']['min_resent_results'][1],
                data['model']['minmax']['min_resent_results'][2],
                data['model']['minmax']['min_resent_results'][3],
                data['model']['minmax']['min_resent_results'][4],
                data['model']['minmax']['min_past_eval'],
                data['model']['minmax']['min_date_hours'],
                data['model']['minmax']['min_max_streak'],
                data['model']['minmax']['min_max_fail_streak'],
                data['model']['minmax']['min_test_count']
    ]

    scaler_max = [
                data['model']['minmax']['max_resent_results'][0],
                data['model']['minmax']['max_resent_results'][1],
                data['model']['minmax']['max_resent_results'][2],
                data['model']['minmax']['max_resent_results'][3],
                data['model']['minmax']['max_resent_results'][4],
                data['model']['minmax']['max_past_eval'],
                data['model']['minmax']['max_date_hours'],
                data['model']['minmax']['max_max_streak'],
                data['model']['minmax']['max_max_fail_streak'],
                data['model']['minmax']['max_test_count']
    ]

    # --------- weight + bias (딥러닝 학습 결과) ---------
    weights = [
                data['model']['weight']['resent_results'][0],
                data['model']['weight']['resent_results'][1],
                data['model']['weight']['resent_results'][2],
                data['model']['weight']['resent_results'][3],
                data['model']['weight']['resent_results'][4],
                data['model']['weight']['past_eval'],
                data['model']['weight']['date_hours'],
                data['model']['weight']['max_streak'],
                data['model']['weight']['max_fail_streak'],
                data['model']['weight']['test_count']
    ]

    bias = data['model']['weight']['bias']

    # feature 벡터 (순서 맞춤)
    feature_vec = [
        resent_results[0],  # recent_0
        resent_results[1],  # recent_1
        resent_results[2],  # recent_2
        resent_results[3],  # recent_3
        resent_results[4],  # recent_4
        past_eval,         # before_predicted_accuracy
        date_hours,        # hours_since_last_quiz
        max_streak,        # streak
        max_fail_streak,   # fail_streak
        test_count         # test
    ]

    # MinMax 스케일링 적용
    scaled_features = [
        (val - min_v) / (max_v - min_v) if (max_v - min_v) > 1e-8 else 0.0
        for val, min_v, max_v in zip(feature_vec, scaler_min, scaler_max)
    ]

    # 선형 결합
    eval = sum(w * x for w, x in zip(weights, scaled_features)) + bias

    eval_prob = sigmoid(eval)
    return round(eval_prob, 3)

# HCE accuracy prediction
def HCE_AP(word_data):
    resent_results = list(map(int, list(word_data['recent_results'])))
    date = (int(time.time()) - int(word_data['last_quiz_timestamp'])) // 3600
    streak = 0
    fail_streak = 0
    past_eval = float(word_data['predicted_accuracy'])
    test_count = int(word_data.get('test', 0))
    for result in resent_results:
        if result == 1:
            streak += 1
            fail_streak = 0
        
        else:
            streak = 0
            fail_streak += 1
    
    w1, w2, w3, w4, w5 = 0.1, 0.15, 0.2, 0.25, 0.3
    w6, w7, w8, w9, w10 = -0.05, 0.4, 0.1, -0.5, 0.05
    
    eval =  resent_results[0] * w1 + \
            resent_results[1] * w2 + \
            resent_results[2] * w3 + \
            resent_results[3] * w4 + \
            resent_results[4] * w5 + \
            date * w6 + \
            streak * w7 + \
            past_eval * w8 + \
            fail_streak * w9 + \
            (test_count - 5) * w10
    
    eval = sigmoid(eval)
    return round(eval, 3)

# 테스트 결과 반영
def reflect_result(result_list):
    word_list = get_wordlist()
    header = word_list[0].keys() if word_list else [] # csv 해더 추출

    with open('user_info/testdata.csv', 'a', newline = '', encoding = 'utf-8') as file: # 딥러닝 학습용 데이터 남기기
        writer = csv.writer(file)

        for word_data in result_list:
            result_line = word_data['recent_results'] # 최근 결과 업데이트
            result_line += ('1' if word_data['judge'] else '0') # 결과 맨 앞에 추가
            word_list[word_data['num'] - 1]['recent_results']             = result_line[1:] # 과거 값 삭제 후 현재 값 적용
            word_list[word_data['num'] - 1]['last_quiz_timestamp']        = int(time.time()) # 테스트 시각
            word_list[word_data['num'] - 1]['test']                       = int(word_data['test']) + 1 # 테스트 갯수 + 1
            word_list[word_data['num'] - 1]['predicted_accuracy']         = (DEEP_AP(word_list[word_data['num'] - 1])
                                                                             if int(word_data['test']) > 4
                                                                             else HCE_AP(word_list[word_data['num'] - 1])) # 예상 정확도 계산 (테스트 횟수가 5 이상이면 딥러닝 예측, 미만이면 HCE 예측)

            
            if int(word_data['test']) > 4:
                row = [
                    ('1' if word_data['judge'] else '0'),
                    word_data['recent_results'],
                    word_data['last_quiz_timestamp'],
                    word_data['predicted_accuracy'],
                    word_data['test']
                ]

                writer.writerow(row)

    with open('user_info/wordlist.csv', 'w', newline = '', encoding = 'utf-8') as file: # 단어 추가
        writer = csv.DictWriter(file, fieldnames = header)
        writer.writeheader()
        writer.writerows(word_list)

# 단어 테스트
def runtest(test_word_list):
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

    result_list = []
    correct = 0
    wrong = 0
    for i, testcase in enumerate(test_word_list, start = 1):
        answer = input(str(i) + '. ' + testcase['word'] + '\n')
        
        judge = False # 정답 판정 (True : 정답, False : 오답)
        for correct_answer in tuple(testcase['mean'].split('|')):
            if is_correct(answer, correct_answer):
                judge = True
                break

        result_list.append( # 결과 리스트에 단어 정보 추가
            {
                'num' : testcase['num'],
                'word' : testcase['word'],
                'mean' : testcase['mean'],
                'test' : testcase['test'],
                'recent_results' : testcase['recent_results'],
                'last_quiz_timestamp' : testcase['last_quiz_timestamp'],
                'predicted_accuracy' : testcase['predicted_accuracy'],
                'judge' : (True if judge else False)
            }
        )

        if judge:
            print(f'{GREEN}Right!{RESET}\nMean : ' + ', '.join(tuple(testcase['mean'].split('|'))))
            correct += 1

        else:
            print(f'{RED}Wrong answer :({RESET}\nMean : ' + ', '.join(tuple(testcase['mean'].split('|'))))
            wrong += 1

    reflect_result(result_list)
    print(f'{CYAN}| Result |{RESET}\nTotal : {correct + wrong}\n{GREEN}Correct : {correct}{RESET}\n{RED}Wrong : {wrong}{RESET}\nCorrect rate : {int((correct / (correct + wrong)) * 100)}%')

# 명령어 함수 -------------------------------------------------->
# 도움말
def help():
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    ansi_escape = re.compile(r'\x1b\[([0-9;]*[a-zA-Z])') #터미널 색상·스타일을 지정하는 ANSI 이스케이프 코드를 찾는 정규식

    def strip_ansi(s):
        return ansi_escape.sub('', s)

    def fmt_cmd(cmd, desc, width = 45): #줄맞춤 변형
        clean_len = len(strip_ansi(cmd))
        pad = width - clean_len #/cmd -.. 이후 | 이전까지 공백 크기 계산

        return f"{cmd}{' ' * pad} | {desc}"

    print(fmt_cmd(f"{GREEN}/help{RESET}", "Show this help message"))
    print(fmt_cmd(f"{GREEN}/addw {BLUE}-w WORD -m MEAN -e \"EXAMPLE\"{RESET}", "Add a new word"))
    print(fmt_cmd(f"{GREEN}/removew {BLUE}-l LIST{RESET}", "Remove words by list numbers"))
    print(fmt_cmd(f"{GREEN}/listw {BLUE}-a(optional) -i(optional) {RESET}", "Show words in wordlist.csv"))
    print(fmt_cmd(f"{GREEN}/randomtest {BLUE}-n NUMBER{RESET}", "Test on random words"))
    print(fmt_cmd(f"{GREEN}/dltest {BLUE}-n NUMBER{RESET}", "Test words selected by DL system"))
    print(fmt_cmd(f"{GREEN}/selectw {BLUE}-l LIST{RESET}", "Select words by list numbers"))
    print(fmt_cmd(f"{GREEN}/unselectw {BLUE}-l LIST{RESET}", "Unselect words by list numbers"))
    print(fmt_cmd(f"{GREEN}/exit{RESET}", "Kill program"))

# 단어 추가
def addw(option): #{'w' : .., 'm' : .., 'w' : ..}
    if not 'w' in option: # 명령어 예외 처리 - w KEY 가 없는 경우
        print('Invalid input: Can\'t find -w option')
        return

    if not 'm' in option: # 명령어 예외 처리 - m KEY 가 없는 경우
        print('Invalid input: Can\'t find -m option')
        return

    if not 'e' in option: # 명령어 예외 처리 - e KEY 가 없는 경우
        print('Invalid input: Can\'t find -e option')
        return

    word_list = get_wordlist()

    for word_data in word_list: # 명령어 예외 처리 - 중복 체크
        if option['w'] in word_data['word']:
            print('Warning: Already exists')
            return

    #단어 추가
    with open('user_info/wordlist.csv', 'a', newline = '', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        row = [
            option['w'],  # 단어
            option['m'],  # 의미
            '0',
            '00000',      # 고정값
            int(time.time()),          # 고정값(unix time)
            '0',          # 고정값
            option['e'],  # 예문
            '0.0'         # 고정값
        ]
        writer.writerow(row)

# 단어 삭제
def removew(option): #{'l' : '1 3 9 ...'}
    if not 'l' in option: # 명령어 예외 처리 - l KEY 가 없는 경웅
        print('Invalid input: Can\'t find -l option')
        return

    word_list = get_wordlist()
    header = word_list[0].keys() if word_list else [] # csv 해더 추출

    new_word_list = [] # 업데이트 할 내용이 들어가는 단어 리스트
    num_set = set(map(int, shlex.split(option['l']))) # 들어가면 안되는 단어 번호 집합

    for i, word_data in enumerate(word_list, start = 1): # 삭제 안 할 단어만 추가
        if not i in num_set:
            new_word_list.append(word_data)

    with open('user_info/wordlist.csv', 'w', newline = '', encoding = 'utf-8') as file: # 단어 추가
        writer = csv.DictWriter(file, fieldnames = header)
        writer.writeheader()
        writer.writerows(new_word_list)

# 랜덤 단어 테스트
def randomtest(option): #{'n' : '12'}
    if not 'n' in option: # 명령어 예외 처리 - n에 값이 없을 경우
        print('Invalid input: Can\'t find -n option')
        return
    
    if not option['n']: # 명령어 예외 처리 - n KEY가 없을 경우
        print('Invalid input: NUMBER is empty')
        return

    option['n'] = int(option['n'])
    
    word_list = get_wordlist()
    selected_word_list = []

    selected_word_total = 0 # 선택된 단어의 갯수
    for i, word_data in enumerate(word_list, start = 1): # 선택된 단어만 selected_word_list에 추가
        if word_data['is_selected'] == '1':
            selected_word_total += 1

            selected_word_list.append( #선택된 단어 정보 추가
                    {
                        'num' : i,
                        'word' : word_data['word'],
                        'mean' : word_data['mean'],
                        'test' : word_data['test'],
                        'recent_results' : word_data['recent_results'],
                        'last_quiz_timestamp' : word_data['last_quiz_timestamp'],
                        'predicted_accuracy' : word_data['predicted_accuracy']
                    }
                )
    

    if option['n'] > selected_word_total: # 명령어 예외 처리 - 선택된 단어의 수를 초과했을 경우
        print('Invalid input: The number of words selected for the test has been exceeded')
        return

    runtest(random.sample(selected_word_list, option['n']))

# 딥러닝 단어 테스트
def dltest(option): #{'n' : '12'}
    if not 'n' in option: # 명령어 예외 처리 - n에 값이 없을 경우
        print('Invalid input: Can\'t find -n option')
        return
    
    if not option['n']: # 명령어 예외 처리 - n KEY가 없을 경우
        print('Invalid input: NUMBER is empty')
        return

    option['n'] = int(option['n'])
    
    word_list = get_wordlist()
    selected_word_list = []

    selected_word_total = 0 # 선택된 단어의 갯수
    for i, word_data in enumerate(word_list, start = 1): # 선택된 단어만 selected_word_list에 추가
        if word_data['is_selected'] == '1':
            selected_word_total += 1

            selected_word_list.append( #선택된 단어 정보 추가
                    {
                        'num' : i,
                        'word' : word_data['word'],
                        'mean' : word_data['mean'],
                        'test' : word_data['test'],
                        'recent_results' : word_data['recent_results'],
                        'last_quiz_timestamp' : word_data['last_quiz_timestamp'],
                        'predicted_accuracy' : word_data['predicted_accuracy']
                    }
                )
    

    if option['n'] > selected_word_total: # 명령어 예외 처리 - 선택된 단어의 수를 초과했을 경우
        print('Invalid input: The number of words selected for the test has been exceeded')
        return

    selected_word_list.sort(key = lambda x : x['predicted_accuracy'])
    runtest(selected_word_list[:option['n']])

# 모든 단어 정보 출력
def listw(option): #{'a' : .., 'i' : ..}
    word_list = get_wordlist()

    if 'a' in option: 
        for word_data in word_list:
            print(
                word_data['word'] + ','
                + word_data['mean'] + ','
                + word_data['recent_results'] + ','
                + word_data['last_quiz_timestamp'] + ','
                + word_data['is_selected'] + ','
                + word_data['example sentence'] + ','
                + word_data['predicted_accuracy']
                )

    if 'i' in option:
        word_total = len(word_list)
        selected_word_total = 0

        for word_data in word_list:
            if word_data['is_selected'] == '1':
                selected_word_total += 1
        
        print(f'Total: {word_total}\nSelected total: {selected_word_total}')

# 단어 출제 등록
def selectw(option): #{'l' : '1 3 9 ...'}
    if not 'l' in option: # 명령어 예외 처리 - l KEY 가 없는 경웅
        print('Invalid input: Can\'t find -l option')
        return

    word_list = get_wordlist()
    header = word_list[0].keys() if word_list else [] # csv 해더 추출

    new_word_list = [] # 업데이트 할 내용이 들어가는 단어 리스트
    num_set = set(map(int, shlex.split(option['l']))) # 변경할 단어 번호 집합

    for i, word_data in enumerate(word_list, start = 1): # 삭제 안 할 단어만 추가
        if i in num_set:
            word_data['is_selected'] = '1'
        
        new_word_list.append(word_data)

    with open('user_info/wordlist.csv', 'w', newline = '', encoding = 'utf-8') as file: # 단어 추가
        writer = csv.DictWriter(file, fieldnames = header)
        writer.writeheader()
        writer.writerows(new_word_list)

# 단어 출제 등록 해제
def unselectw(option): #{'l' : '1 3 9 ...'}
    if not 'l' in option: # 명령어 예외 처리 - l KEY 가 없는 경웅
        print('Invalid input: Can\'t find -l option')
        return

    word_list = get_wordlist()
    header = word_list[0].keys() if word_list else [] # csv 해더 추출

    new_word_list = [] # 업데이트 할 내용이 들어가는 단어 리스트
    num_set = set(map(int, shlex.split(option['l']))) # 변경할 단어 번호 집합

    for i, word_data in enumerate(word_list, start = 1): # 삭제 안 할 단어만 추가
        if i in num_set:
            word_data['is_selected'] = '0'
        
        new_word_list.append(word_data)

    with open('user_info/wordlist.csv', 'w', newline = '', encoding = 'utf-8') as file: # 단어 추가
        writer = csv.DictWriter(file, fieldnames = header)
        writer.writeheader()
        writer.writerows(new_word_list)

# 메인 함수 ---------------------------------------------------->
def main():
    # ANSI 색상 코드
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    print(
f'''{CYAN}=========================================
DVOCA  —  Deep Learning Vocabulary
=========================================
{RESET}{YELLOW}Version:{RESET} 2.0.0
{YELLOW}Author:{RESET} Han Junsu
{YELLOW}License:{RESET} Apache License 2.0 — {GREEN}Free for commercial use, modification, and distribution{RESET}

Open source deep learning vocabulary project.

----- manual -----
All commands start with '/'
'''
    )
    
    help()

    # 메인 루프
    while True:
        cmd, option = parse_command(input())

        if cmd == '/help':
            help()

        elif cmd == '/addw':
            addw(option)

        elif cmd == '/removew':
            removew(option)

        elif cmd == '/listw':
            listw(option)

        elif cmd == '/randomtest':
            randomtest(option)

        elif cmd == '/dltest':
            dltest(option)

        elif cmd == '/selectw':
            selectw(option)

        elif cmd == '/unselectw':
            unselectw(option)

        elif cmd == '/exit':
            exit()
        
        else:
            print('Invalid input: Invalid \'/\' command')

main()