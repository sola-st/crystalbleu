import re
import json
import random
from collections import Counter
from CodeBLEU.code_bleu import code_bleu
from bleu_ignoring import sentence_bleu, SmoothingFunction
from nltk.util import ngrams
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CppLexer

MAXN=4

sm_func = SmoothingFunction(epsilon=0.0001).method1
lexer = JavaLexer()
# lexer = CppLexer()

all_ngrams = []
with open('lang2.json') as f:
    data = json.load(f)
print(len(data.items()))
for k, v in data.items():
    prob_name = k.split('_')
    if int(prob_name[0]) > 1 or len(prob_name[1]) > 4:
        continue
    for i in v:
        if random.random() < 0.3:
            temp = list(map(lambda x: x[1], lexer.get_tokens(i)))
            tokenized = []
            for tok in temp:
                if (not re.fullmatch('\s+', tok)) and (not re.fullmatch('\/\/.*\n', tok)) and (not re.fullmatch('\/\*.*\*\/', tok, re.DOTALL)):
                    tokenized.append(tok)
            for j in range(1, MAXN+1):
                n_grams = list(ngrams(tokenized, j))
                all_ngrams.extend(n_grams)
freq = Counter(all_ngrams)

# # ref:
# code1 = "#include <iostream>\n\
# #include <string>\n\
# #include <stdio.h>\n\
# using namespace std;\n\
# int main()\n\
# {\n\
# 	int i, j;\n\
# 	string s, v, t;\n\
# 	getline(cin,s);\n\
# 	getline(cin,v);\n\
# 	cout << v << endl << s << endl;\n\
# 	while ( getline(cin,t) ){\n\
# 		for ( i = 0; i < t.size(); i++ ){\n\
# 			for ( j = 0; j < s.size(); j++ ){\n\
# 				if ( t[i] == s[j] ){\n\
# 					t[i] = v[j];\n\
# 					break;\n\
# 				}\n\
# 			}\n\
# 		}\n\
# 		cout << t << endl;\n\
# 	}\n\
# }\n\
# "
# # hyp1:
# code2 = "#include <iostream>\n\
# #include <string>\n\
# using namespace std;\n\
# int main(){\n\
# 	string a,b,s,t,n;\n\
# 	n  = \"\\n\";\n\
# 	getline(cin,a);\n\
# 	getline(cin,b);\n\
# 	cout<<b+n+a+n;\n\
# 	while(getline(cin,s))\n\
# 	{\n\
# 		t=\"\";\n\
# 		for(int i = 0 ;i <s.length();i++)\n\
# 		{\n\
# 		    int d=a.find(s[i]);\n\
# 			if(d!=string::npos)\n\
# 				t += b[d];\n\
# 			else\n\
# 				t+=s[i];\n\
# 		}\n\
# 		cout<<t+n;\n\
# 	}\n\
# }\n\
# "
# # hyp2:
# code3 = "#include<iostream>\n\
# #include<vector>\n\
# using namespace std;\n\
# int main()\n\
# {\n\
#     int a,b,c;\n\
#     vector <int> N;\n\
#     while(cin>>a>>b)\n\
#     {\n\
#         N.push_back(a);\n\
#         N.push_back(b);\n\
#     }\n\
#     c=N.size();\n\
#     for(int i=0;i<c/2;i++)\n\
#         cout<<N[2*i]+N[2*i+1]<<endl;\n\
# }\n\
# "

# ref:
code1 = "import java.util.*;\
public class Main {\
    public static void main(String[] args) {\
        Scanner i = new Scanner(System.in);\
        int t = i.nextInt();\
        i.nextLine();\
        while (t-- > 0) {\
            System.out.println(new StringBuffer(i.nextLine()).reverse());\
        }\
    }\
}\
"

# hyp1:
code2 = "import java.util.Scanner;\
public class Main {\
	public static void main(String argv[]) {\
		int num_of_tests = 0;\
		Scanner in = new Scanner(System.in);\
		num_of_tests = Integer.parseInt(in.nextLine());\
		for(int i=0; i<num_of_tests; i++) {\
			String str = in.nextLine();\
			StringBuilder rev_str = new StringBuilder();\
			rev_str.append(str);\
			System.out.println(rev_str.reverse());\
		}\
	}\
}\
"

# hyp2:
code3 = "import java.util.Scanner;\
public class Main {\
    public static void main(String[] args) {\
        Scanner cin = new Scanner(System.in);\
        while (cin.hasNext())\
            System.out.println(cin.nextInt() + cin.nextInt());\
    }\
}\
"

# code1 = "import java.util.Scanner;\
# public class Main {\
#     public static void main(String[] args) {\
#         Scanner in = new Scanner(System.in);\
#         while (in.hasNext()) {\
#             int a = in.nextInt();\
#             int x = 2;\
#             while (x <= a) {\
#                 if (a % x == 0)\
#                     break;\
#             x += 1;\
#             }\
#             System.out.println((x >= a)?”Prime”:”Not prime”);\
#         }\
#     }\
# }\
# "

# code2 = "import java.util.Scanner;\
# public class Main {\
#     public static void main(String[] args) {\
#         Scanner in = new Scanner(System.in);\
#         while (in.hasNextInt()) {\
#             int a = in.nextInt();\
#             int b = in.nextInt();\
#             System.out.println(a + b);\
#         }\
#     }\
# }\
# "

# code3 = "import java.util.*;\
# public class Main {\
#     public static void main(String[] args){\
#         int x;\
#         Scanner in = new Scanner(System.in);\
#         while (true) {\
#             if (!in.hasNext())\
#                 break;\
#             x = in.nextInt();\
#             x += in.nextInt();\
#             System.out.println(x);\
#         }\
#     }\
# }\
# "

tokenized1 = [t for t in list(map(lambda x: x[1], lexer.get_tokens(code1))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))]
tokenized2 = [t for t in list(map(lambda x: x[1], lexer.get_tokens(code2))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))]
tokenized3 = [t for t in list(map(lambda x: x[1], lexer.get_tokens(code3))) if not (re.fullmatch('\s+', t) or re.fullmatch('\/\/.*\n', t) or re.match('\/\*.*\*\/', t, re.DOTALL))]

print(tokenized1, tokenized2, tokenized3)
print(len(tokenized1), len(tokenized2), len(tokenized3))

most_common_dict = dict(freq.most_common(200))

print(sentence_bleu([tokenized1], tokenized3, smoothing_function=sm_func, ignoring=None))
print(sentence_bleu([tokenized1], tokenized3, smoothing_function=sm_func, ignoring=most_common_dict))

print(sentence_bleu([tokenized1], tokenized2, smoothing_function=sm_func, ignoring=None))
print(sentence_bleu([tokenized1], tokenized2, smoothing_function=sm_func, ignoring=most_common_dict))

print(code_bleu([[tokenized1]], [tokenized3]), code_bleu([[tokenized1]], [tokenized2]))