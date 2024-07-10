

system_msg = """You will be provided with some examples (delimited with tags <example> and </example>). In each of the example, there are three parts. (1) the query string (delimited with tags <q> and </q>). (2) some positive strings (each of them is delimited with tags <p> and </p>). (3) some negative strings (each of them is delimited with tags <n> and </n>). Your task is divided into two steps. Step 1: For each example, find common information that appears in query string and all of the positive strings. All strings are about 3 digit addition, such as "1 2 8 + 4 0 9 = 5 3 7". The digits in the first operand is referred to as F1, F2, F3, digits in the second operand is referred to as S1, S2, S3, digits in answer is referred to as A1, A2, A3, A4 (if exsits). The carry produced by the sum of F2 and S2 is referred to as C2, in other words, C2 means whether F2+S2>=10. Likewise, C3 means whether F3+S3>=10. In the above example, F1=1, F2=2, F3=8, S1=4, S2=0, S3=9, A1=5, A2=3, A3=7. The common information should be described using these name, and should be consistent in query string and all the positive strings. Negative strings may share part of the common information, but not all of it. Because those strings which have this common information are positive while those which do not have are negative strings. In all strings, only part of the string is in the parenthesis, which is called "visible part" you should only look for common information in visible part. The common information may be same digits appearing in the same place (in terms of hundreds / tens / ones place), or digits in a certain place are always in a certain range, or the sum of the two digits from the same place is always in a certain range, etc. "+" and "=" should not be considered as common information. You should output the common information (delimited with tags <info> and </info>), and the index of the corresponding example. If no positive strings available, you can answer "Unknown" or "Uncertain". Step 2: After finding all the common information from examples, you should summarize your findings across different examples and output a succinct description (delimited with tags <summary> and </summary>)."""

messages=[{"role": "system", "content": system_msg}]

# user_msg = make_user_msg("blocks.0.attn.hook_result.0", 8)

user_msg = """<example> index: 0 
<q> ( 8 5 9 + 1 3 7 = ) 9 9 6 </q>
<p> ( 8 2 1 + 1 4 2 = ) 9 6 3  </p>
<p> ( 8 2 3 + 1 3 2 = ) 9 5 5  </p>
<p> ( 8 9 2 + 1 7 9 = ) 1 0 7 1  </p>
<p> ( 8 9 4 + 1 8 1 = ) 1 0 7 5  </p>
<p> ( 8 3 3 + 1 4 4 = ) 9 7 7  </p>
<p> ( 8 9 0 + 1 8 9 = ) 1 0 7 9  </p>
<p> ( 8 1 0 + 1 0 4 = ) 9 1 4  </p>
<p> ( 1 2 7 + 8 7 0 = ) 9 9 7  </p>
<p> ( 1 8 6 + 8 7 5 = ) 1 0 6 1  </p>
<p> ( 1 3 2 + 8 2 0 = ) 9 5 2  </p>
<p> ( 1 0 4 + 8 3 9 = ) 9 4 3  </p>
<p> ( 1 4 7 + 8 0 0 = ) 9 4 7  </p>
<p> ( 1 7 2 + 8 4 4 = ) 1 0 1 6  </p>
<p> ( 1 8 1 + 8 0 2 = ) 9 8 3  </p>
<p> ( 1 1 0 + 8 4 7 = ) 9 5 7  </p>
<n> ( 7 6 7 + 1 5 8 = ) 9 2 5  </n>
<n> ( 7 0 9 + 1 3 2 = ) 8 4 1  </n>
<n> ( 7 0 9 + 1 5 5 = ) 8 6 4  </n>
<n> ( 7 2 0 + 1 6 9 = ) 8 8 9  </n>
<n> ( 7 3 2 + 1 0 0 = ) 8 3 2  </n>
<n> ( 7 9 3 + 1 2 4 = ) 9 1 7  </n>
<n> ( 1 5 5 + 7 7 4 = ) 9 2 9  </n>
<n> ( 1 8 6 + 7 5 4 = ) 9 4 0  </n>
<n> ( 0 0 7 + 8 0 5 = ) 1 9 1 2  </n>
<n> ( 9 8 6 + 1 3 3 = ) 1 1 1 9  </n>
</example>

<example> index: 1 
<q> ( 3 0 8 + 1 1 8 = ) 4 2 6 </q>
<p> ( 3 0 1 + 1 7 5 = ) 4 7 6  </p>
<p> ( 3 6 8 + 1 7 1 = ) 5 3 9  </p>
<p> ( 3 4 0 + 1 8 9 = ) 5 2 9  </p>
<p> ( 3 4 8 + 1 7 1 = ) 5 1 9  </p>
<p> ( 3 4 1 + 1 2 1 = ) 4 6 2  </p>
<p> ( 3 0 5 + 1 3 0 = ) 4 3 5  </p>
<p> ( 1 1 1 + 3 7 9 = ) 4 9 0  </p>
<p> ( 1 2 2 + 3 3 2 = ) 4 5 4  </p>
<p> ( 1 2 8 + 3 3 1 = ) 4 5 9  </p>
<p> ( 1 2 5 + 3 9 7 = ) 5 2 2  </p>
<p> ( 1 9 3 + 3 0 6 = ) 4 9 9  </p>
<p> ( 1 5 5 + 3 3 7 = ) 4 9 2  </p>
<p> ( 1 0 7 + 3 1 4 = ) 4 2 1  </p>
<p> ( 1 4 2 + 3 2 4 = ) 4 6 6  </p>
<p> ( 1 0 4 + 3 0 4 = ) 4 0 8  </p>
<p> ( 1 8 1 + 3 4 8 = ) 5 2 9  </p>
<p> ( 4 2 7 + 1 6 8 = ) 5 9 5  </p>
<p> ( 4 4 1 + 1 2 6 = ) 5 6 7  </p>
<p> ( 4 7 9 + 1 2 4 = ) 6 0 3  </p>
<p> ( 1 7 3 + 1 0 9 = ) 2 8 2  </p>
<n> ( 2 7 3 + 1 6 3 = ) 4 3 6  </n>
<n> ( 2 4 3 + 1 8 7 = ) 4 3 0  </n>
<n> ( 2 7 1 + 1 8 2 = ) 4 5 3  </n>
<n> ( 6 7 3 + 1 4 6 = ) 8 1 9  </n>
<n> ( 4 8 7 + 2 7 2 = ) 7 5 9  </n>
</example>
"""

messages.append({"role": "user", "content": user_msg})


# assistant_msg = """
# <info> index: 0; The visible part in query string is "8 5 9 + 1 3 7 =". In the visible part of positive strings, "8" and "1" always appear also in hundreds place of the two operands. On the contrary, in the visible part of negative strings, either "8" or "1" may appear alone, but never together. Therefore, the information that query and all positive strings have in common is F1 and S1 (order may change). </info>
# <info> index: 1; The visible part in query string is "3 0 8 + 1 1 8 =". In the visible part of positive strings, "1" always appears also in hundreds place of one operand. The other digit in the hundreds place of the query stirng, "3", is also approximated in positive string. Therefore, the information that query and all positive strings have in common is S1. </info>
# <summary> 1-2 digits from F1 and S1 </summary>
# """

assistant_msg = """
<info> index: 0; The visible part in query string is "8 5 9 + 1 3 7 =". The information that query and all positive strings have in common is F1 ("8") and S1 ("1") (order may change). </info>
<info> index: 1; The visible part in query string is "3 0 8 + 1 1 8 =". The information that query and all positive strings have in common is S1 ("1"). </info>
<summary> 1-2 digits from F1 and S1 </summary>
"""

messages.append({"role": "assistant", "content": assistant_msg})


# user_msg = make_user_msg("blocks.0.hook_resid_pre", 9)

user_msg = """<example> index: 0 
<q> ( 8 3 7 + 9 3 6 = 1 ) 7 7 3 </q>
<p> ( 7 1 3 + 3 1 9 = 1 ) 0 3 2  </p>
<p> ( 3 9 9 + 9 8 6 = 1 ) 3 8 5  </p>
<p> ( 7 6 1 + 7 0 4 = 1 ) 4 6 5  </p>
<p> ( 9 9 2 + 7 2 2 = 1 ) 7 1 4  </p>
<p> ( 7 3 8 + 4 3 0 = 1 ) 1 6 8  </p>
<p> ( 8 2 0 + 5 9 4 = 1 ) 4 1 4  </p>
<p> ( 5 1 7 + 6 4 5 = 1 ) 1 6 2  </p>
<p> ( 4 6 2 + 8 4 2 = 1 ) 3 0 4  </p>
<p> ( 6 7 4 + 4 0 4 = 1 ) 0 7 8  </p>
<p> ( 7 2 7 + 3 4 7 = 1 ) 0 7 4  </p>
<p> ( 9 3 6 + 3 0 2 = 1 ) 2 3 8  </p>
<p> ( 5 3 0 + 9 3 2 = 1 ) 4 6 2  </p>
<p> ( 6 9 1 + 6 4 2 = 1 ) 3 3 3  </p>
<p> ( 9 6 8 + 4 0 2 = 1 ) 3 7 0  </p>
<p> ( 5 7 7 + 7 7 7 = 1 ) 3 5 4  </p>
<p> ( 4 1 7 + 8 6 7 = 1 ) 2 8 4  </p>
<p> ( 8 5 6 + 4 6 8 = 1 ) 3 2 4  </p>
<p> ( 3 7 8 + 9 5 6 = 1 ) 3 3 4  </p>
<p> ( 9 4 0 + 2 9 5 = 1 ) 2 3 5  </p>
<p> ( 9 8 2 + 4 8 8 = 1 ) 4 7 0  </p>
<n> ( 4 8 6 + 4 0 6 = ) 8 9 2  </n>
<n> ( 1 3 3 + 6 9 4 = ) 8 2 7  </n>
<n> ( 8 1 7 + 1 7 2 = ) 9 8 9  </n>
<n> ( 1 6 5 + 7 7 0 = ) 9 3 5  </n>
<n> ( 4 3 0 + 5 1 8 = ) 9 4 8  </n>
</example>

<example> index: 1 
<q> ( 2 7 0 + 3 4 8 = 6 ) 1 8 </q>
<p> ( 1 3 7 + 5 0 6 = 6 ) 4 3  </p>
<p> ( 4 2 1 + 2 5 7 = 6 ) 7 8  </p>
<p> ( 2 1 4 + 4 1 7 = 6 ) 3 1  </p>
<p> ( 3 6 9 + 3 0 1 = 6 ) 7 0  </p>
<p> ( 4 5 7 + 2 0 9 = 6 ) 6 6  </p>
<p> ( 4 2 8 + 2 5 4 = 6 ) 8 2  </p>
<p> ( 4 2 1 + 2 5 0 = 6 ) 7 1  </p>
<p> ( 2 7 6 + 3 8 7 = 6 ) 6 3  </p>
<p> ( 1 9 6 + 4 1 8 = 6 ) 1 4  </p>
<p> ( 1 0 9 + 5 7 3 = 6 ) 8 2  </p>
<p> ( 1 3 8 + 5 1 1 = 6 ) 4 9  </p>
<p> ( 2 5 7 + 4 2 4 = 6 ) 8 1  </p>
<p> ( 1 7 0 + 4 3 0 = 6 ) 0 0  </p>
<p> ( 3 6 3 + 2 7 9 = 6 ) 4 2  </p>
<p> ( 1 3 3 + 5 0 4 = 6 ) 3 7  </p>
<p> ( 2 2 9 + 4 5 3 = 6 ) 8 2  </p>
<p> ( 2 8 2 + 3 2 3 = 6 ) 0 5  </p>
<p> ( 3 3 6 + 3 1 6 = 6 ) 5 2  </p>
<n> ( 9 0 4 + 7 8 5 = 1 6 ) 8 9  </n>
<n> ( 4 4 9 + 1 1 6 = 5 6 ) 5  </n>
<n> ( 1 2 3 + 4 2 9 = 5 ) 5 2  </n>
<n> ( 4 5 3 + 1 3 1 = 5 ) 8 4  </n>
<n> ( 2 5 7 + 4 7 8 = 7 ) 3 5  </n>
<n> ( 2 5 1 + 4 9 3 = 7 ) 4 4  </n>
<n> ( 4 0 7 + 2 9 3 = 7 ) 0 0  </n>
</example>
"""

messages.append({"role": "user", "content": user_msg})


# assistant_msg = """
# <info> index: 0; The visible part in query string is "8 3 7 + 9 3 6 = 1". In the visible part of positive strings, A1 ("1") is always the same as the A1 ("1") in query string. On the contrary, in the visible part of negative strings, A1 is different from the query. Therefore, the information that query and all positive strings have in common is A1 </info>
# <info> index: 0; The visible part in query string is "2 7 0 + 3 4 8 = 6". In the visible part of positive strings, A1 ("6") is always the same as the A1 ("6") in query string. On the contrary, in the visible part of negative strings, A1 is different from the query. Therefore, the information that query and all positive strings have in common is A1 </info>
# <summary> A1 </summary>
# """

assistant_msg = """
<info> index: 0; The visible part in query string is "8 3 7 + 9 3 6 = 1". The information that query and all positive strings have in common is A1 ("1") </info>
<info> index: 0; The visible part in query string is "2 7 0 + 3 4 8 = 6". The information that query and all positive strings have in common is A1 ("6") </info>
<summary> A1 </summary>
"""

messages.append({"role": "assistant", "content": assistant_msg})

# user_msg = make_user_msg("blocks.1.attn.hook_result.0", 9, filter_func=lambda x: x[9]=="1")

user_msg = """<example> index: 0 
<q> ( 3 7 9 + 7 0 1 = 1 ) 0 8 0 </q>
<p> ( 3 6 7 + 7 0 0 = 1 ) 0 6 7  </p>
<p> ( 7 5 6 + 3 0 1 = 1 ) 0 5 7  </p>
<p> ( 6 3 9 + 4 5 2 = 1 ) 0 9 1  </p>
<p> ( 4 5 9 + 6 3 3 = 1 ) 0 9 2  </p>
<p> ( 4 3 8 + 6 5 2 = 1 ) 0 9 0  </p>
<p> ( 7 1 2 + 3 6 9 = 1 ) 0 8 1  </p>
<p> ( 7 5 8 + 3 2 9 = 1 ) 0 8 7  </p>
<p> ( 4 7 3 + 6 1 2 = 1 ) 0 8 5  </p>
<p> ( 7 8 0 + 3 0 0 = 1 ) 0 8 0  </p>
<p> ( 4 2 2 + 6 6 6 = 1 ) 0 8 8  </p>
<p> ( 4 0 9 + 6 8 5 = 1 ) 0 9 4  </p>
<p> ( 7 4 6 + 3 3 6 = 1 ) 0 8 2  </p>
<p> ( 7 4 8 + 3 3 3 = 1 ) 0 8 1  </p>
<p> ( 4 2 6 + 6 6 6 = 1 ) 0 9 2  </p>
<p> ( 7 5 9 + 3 3 8 = 1 ) 0 9 7  </p>
<p> ( 2 7 1 + 8 1 4 = 1 ) 0 8 5  </p>
<p> ( 7 2 3 + 3 6 6 = 1 ) 0 8 9  </p>
<n> ( 8 4 9 + 2 3 0 = 1 ) 0 7 9  </n>
<n> ( 7 4 5 + 3 3 3 = 1 ) 0 7 8  </n>
<n> ( 7 3 5 + 3 4 4 = 1 ) 0 7 9  </n>
<n> ( 4 4 6 + 6 2 6 = 1 ) 0 7 2  </n>
<n> ( 7 1 5 + 3 3 4 = 1 ) 0 4 9  </n>
<n> ( 8 1 5 + 2 3 8 = 1 ) 0 5 3  </n>
<n> ( 9 1 4 + 2 0 1 = 1 ) 1 1 5  </n>
<n> ( 1 5 1 + 9 2 2 = 1 ) 0 7 3  </n>
</example>

<example> index: 1 
<q> ( 2 8 1 + 8 4 6 = 1 ) 1 2 7 </q>
<p> ( 2 6 0 + 8 8 4 = 1 ) 1 4 4  </p>
<p> ( 2 4 4 + 8 8 7 = 1 ) 1 3 1  </p>
<p> ( 8 6 0 + 2 8 4 = 1 ) 1 4 4  </p>
<p> ( 8 5 0 + 2 8 5 = 1 ) 1 3 5  </p>
<p> ( 2 8 3 + 8 7 3 = 1 ) 1 5 6  </p>
<p> ( 2 6 2 + 8 8 5 = 1 ) 1 4 7  </p>
<p> ( 2 8 6 + 8 4 1 = 1 ) 1 2 7  </p>
<p> ( 8 5 4 + 2 8 8 = 1 ) 1 4 2  </p>
<p> ( 2 5 6 + 8 8 3 = 1 ) 1 3 9  </p>
<p> ( 2 8 8 + 8 6 1 = 1 ) 1 4 9  </p>
<p> ( 8 7 4 + 2 7 9 = 1 ) 1 5 3  </p>
<p> ( 2 7 4 + 8 6 3 = 1 ) 1 3 7  </p>
<p> ( 8 7 6 + 2 6 8 = 1 ) 1 4 4  </p>
<p> ( 2 7 8 + 8 5 7 = 1 ) 1 3 5  </p>
<p> ( 8 6 8 + 2 7 1 = 1 ) 1 3 9  </p>
<p> ( 8 5 2 + 2 7 8 = 1 ) 1 3 0  </p>
<p> ( 2 9 4 + 8 4 4 = 1 ) 1 3 8  </p>
<p> ( 2 6 2 + 8 9 4 = 1 ) 1 5 6  </p>
<p> ( 8 9 2 + 2 2 7 = 1 ) 1 1 9  </p>
<n> ( 8 9 3 + 2 1 4 = 1 ) 1 0 7  </n>
<n> ( 6 3 4 + 4 7 1 = 1 ) 1 0 5  </n>
<n> ( 3 9 7 + 7 9 9 = 1 ) 1 9 6  </n>
<n> ( 1 3 9 + 9 6 3 = 1 ) 1 0 2  </n>
<n> ( 2 6 0 + 9 6 5 = 1 ) 2 2 5  </n>
<n> ( 7 3 8 + 2 9 8 = 1 ) 0 3 6  </n>
</example>

<example> index: 2 
<q> ( 7 4 7 + 9 4 9 = 1 ) 6 9 6 </q>
<p> ( 9 4 6 + 7 4 1 = 1 ) 6 8 7  </p>
<p> ( 9 6 1 + 7 2 8 = 1 ) 6 8 9  </p>
<p> ( 9 1 7 + 7 7 2 = 1 ) 6 8 9  </p>
<p> ( 9 7 4 + 7 1 0 = 1 ) 6 8 4  </p>
<p> ( 7 9 3 + 8 1 0 = 1 ) 6 0 3  </p>
<p> ( 9 0 0 + 7 8 6 = 1 ) 6 8 6  </p>
<p> ( 9 4 4 + 7 5 0 = 1 ) 6 9 4  </p>
<p> ( 7 9 5 + 8 1 9 = 1 ) 6 1 4  </p>
<p> ( 7 9 8 + 8 1 6 = 1 ) 6 1 4  </p>
<p> ( 8 7 0 + 7 3 4 = 1 ) 6 0 4  </p>
<p> ( 7 2 4 + 8 8 2 = 1 ) 6 0 6  </p>
<p> ( 7 7 5 + 8 3 3 = 1 ) 6 0 8  </p>
<p> ( 7 7 1 + 8 3 8 = 1 ) 6 0 9  </p>
<p> ( 8 2 4 + 7 8 8 = 1 ) 6 1 2  </p>
<p> ( 7 2 7 + 8 8 9 = 1 ) 6 1 6  </p>
<p> ( 9 0 1 + 7 0 9 = 1 ) 6 1 0  </p>
<p> ( 7 2 6 + 8 8 8 = 1 ) 6 1 4  </p>
<p> ( 7 4 8 + 9 0 3 = 1 ) 6 5 1  </p>
<n> ( 7 6 3 + 9 3 6 = 1 ) 6 9 9  </n>
<n> ( 9 7 2 + 7 2 4 = 1 ) 6 9 6  </n>
<n> ( 8 8 4 + 7 1 4 = 1 ) 5 9 8  </n>
<n> ( 7 9 2 + 8 0 9 = 1 ) 6 0 1  </n>
<n> ( 8 6 4 + 7 7 2 = 1 ) 6 3 6  </n>
<n> ( 7 1 7 + 8 8 9 = 1 ) 6 0 6  </n>
<n> ( 7 5 6 + 8 1 0 = 1 ) 5 6 6  </n>
</example>
"""

messages.append({"role": "user", "content": user_msg})


# assistant_msg = """
# <info> index: 0; The visible part in query string is "3 7 9 + 7 0 1 = 1". In the visible part of positive strings, F1 and S1 are always one of (3, 7), (4, 6), (2, 8). So an approximate version of F1 and S1 of the query string can be observed in positive strings. In addition, the sum of F2 and S2 is always less than 10 in query and positive strings, so C2 is also kept. On the contrary, in the visible part of negative strings, "9" sometimes appears in hundreds place. Therefore, the information that query and all positive strings have in common is F1 and S1 (fuzzy representation, order may change); C2. </info>
# <info> index: 1; The visible part in query string is "2 8 1 + 8 4 6 = 1". In the visible part of positive strings, F1 and S1 are always (2, 8) (order may change). In addition, in query and all positive strings, F2+S2>=10 also holds. On the contrary, some negtive strings have different F1 and S1. Therefore, the information that query and all positive strings have in common is F1 ans S1 (order may change); C2. </info>
# <info> index: 2; The visible part in query string is "7 4 7 + 9 4 9 = 1". In the visible part of positive strings, "7" always appears as one of F1 or S1, and the other digit is "8" or "9". So an approximate version of F1 and S1 of the query string can be observed in positive strings. Therefore, the information that query and all positive strings have in common is F1 ans S1 (fuzzy representation, order may change). </info>
# <summary> F1 and S1 (sometimes fuzzy); C2 (sometimes) </summary>
# """

assistant_msg = """
<info> index: 0; The visible part in query string is "3 7 9 + 7 0 1 = 1". In the visible part of positive strings, F1 and S1 are always one of (3, 7), (4, 6), (2, 8). So an approximate version of F1 and S1 of the query string can be observed in positive strings. In addition, the sum of F2 and S2 is always less than 10 in query and positive strings, so C2 is also kept. Therefore, the information that query and all positive strings have in common is F1 and S1 (fuzzy representation, order may change); C2. </info>
<info> index: 1; The visible part in query string is "2 8 1 + 8 4 6 = 1". In the visible part of positive strings, F1 and S1 are always (2, 8) (order may change). In addition, in query and all positive strings, F2+S2>=10 also holds. Therefore, the information that query and all positive strings have in common is F1 ans S1 (order may change); C2. </info>
<info> index: 2; The visible part in query string is "7 4 7 + 9 4 9 = 1". In the visible part of positive strings, "7" always appears as one of F1 or S1, and the other digit is "8" or "9". So an approximate version of F1 and S1 of the query string can be observed in positive strings. Therefore, the information that query and all positive strings have in common is F1 ans S1 (fuzzy representation, order may change). </info>
<summary> F1 and S1 (sometimes fuzzy); C2 (sometimes) </summary>
"""

messages.append({"role": "assistant", "content": assistant_msg})
