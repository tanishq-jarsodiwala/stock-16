import pandas as pd
import random
import time
import os

# Load the dataset
data = pd.read_csv("faq_stock_market_platform.csv")

# Clean the dataset: Remove header row and rename columns for clarity
data_cleaned = data.iloc[1:].rename(columns={"Column1": "Question", "Column2": "Answer"}).reset_index(drop=True)

# Filter dataset to include only stock-market-related questions
# This filter checks for keywords related to the stock market
stock_keywords = ["stock", "market", "shares", "trading", "investment", "price", "portfolio", "dividend"]
data_cleaned = data_cleaned[data_cleaned['Question'].str.contains(
    "|".join(stock_keywords), case=False, na=False)]

# Function to generate plausible incorrect answers
def generate_incorrect_answers(correct_answer, all_answers, n=3):
    incorrect = random.sample([ans for ans in all_answers if ans != correct_answer], n)
    return incorrect

# Preparing the 50 questions with answers
all_answers = data_cleaned["Answer"].tolist()
questions = data_cleaned.sample(50, random_state=42).reset_index(drop=True)

# Add MCQ options for each question
quiz_data = []
for i, row in questions.iterrows():
    correct_answer = row["Answer"]
    incorrect_answers = generate_incorrect_answers(correct_answer, all_answers)
    options = incorrect_answers + [correct_answer]
    random.shuffle(options)  # Shuffle to randomize option order
    quiz_data.append({
        "Question": row["Question"],
        "Options": options,
        "Correct": correct_answer
    })

# Gamified quiz function
def gamified_quiz(quiz_data):
    print("\n🎉 Welcome to the Stock Market Quiz Game! 🎉\n")
    print("🌟 Test your knowledge of the stock market and climb the levels to become a Stock Market Guru! 🌟\n")
    levels = 5
    questions_per_level = 10
    score = 0
    emojis = ["💸", "📈", "📉", "💹", "🤑"]
    
    for level in range(1, levels + 1):
        print(f"\n--- 🚀 Level {level} 🚀 ---\n")
        level_questions = quiz_data[(level - 1) * questions_per_level: level * questions_per_level]
        level_score = 0

        for idx, q in enumerate(level_questions, start=1):
            print(f"{emojis[random.randint(0, len(emojis) - 1)]} Q{idx}: {q['Question']}")
            for i, option in enumerate(q['Options'], start=1):
                print(f"  {i}. {option}")
            
            print("\n⏳ You have 25 seconds to answer! ⏳")
            start_time = time.time()
            try:
                answer = int(input("\nEnter your choice (1-4): "))
                elapsed_time = time.time() - start_time

                if elapsed_time > 25:
                    print("⏱ Time's up! Moving to the next question.")
                elif q['Options'][answer - 1] == q['Correct']:
                    print(f"✅ Correct! Well done! {random.choice(['Keep going!', 'You’re on fire!', 'Awesome!'])}")
                    level_score += 1
                else:
                    print(f"❌ Incorrect! The correct answer was: {q['Correct']}. Don’t worry, you’ll get the next one!")
            except (ValueError, IndexError):
                print(f"❌ Invalid input! The correct answer was: {q['Correct']}")

            print("\n" + "📊 Progress: " + "▮" * idx + "-" * (questions_per_level - idx) + "\n")
            time.sleep(1)  # Short pause for engagement
        
        score += level_score
        print(f"🎉 Level {level} complete! You scored {level_score}/{questions_per_level}.\n")
        if level_score < questions_per_level // 2:
            print("😔 You need to score more to pass this level. Better luck next time!")
            break
        else:
            print(f"🎯 Great job! Get ready for Level {level + 1 if level < levels else 'Mastery'}!")

    print(f"\n🏆 Quiz Over! Your total score is: {score}/{levels * questions_per_level}")
    if score == levels * questions_per_level:
        print("🎊 Congratulations! You are a Stock Market Master! 🎊")
    elif score > (levels * questions_per_level // 2):
        print("💡 Good effort! Keep learning and you’ll master the stock market soon.")
    else:
        print("📚 Don’t give up! Brush up on your stock market knowledge and try again.")

# Run the quiz
gamified_quiz(quiz_data)
