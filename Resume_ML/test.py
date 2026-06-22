from resume_matcher import match_resume

job_description = """
Python
SQL
Machine Learning
Data Analysis
"""

name, score = match_resume(job_description)

print("Best Resume:", name)
print("Match Score:", score)