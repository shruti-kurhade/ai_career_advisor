import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🎯",
    layout="centered"
)

# ----------------------------
# PROFESSIONAL CSS
# ----------------------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Main container */
.main {
    background-color: #0f172a;
}

/* Title */
h1 {
    color: #38bdf8;
    text-align: center;
    font-weight: 800;
}

/* Text */
p, label {
    color: #94a3b8 !important;
    font-size: 16px;
    font-weight:500;
}

/* Text area */
textarea {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #1e293b !important;
    color: white !important;
    padding: 10px;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
    font-weight: 800;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
}

/* Table styling */
table {
    border-radius: 12px;
    overflow: hidden;
}

/* Career card style */
div[data-testid="stTable"] {
    background-color: #1e293b;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# SAMPLE DATASET

career_data = {
    "Career": [
        "Data Scientist",
        "Machine Learning Engineer",
        "AI Engineer",
        "Data Analyst",
        "Business Analyst",
        "Web Developer",
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "UI/UX Designer",
        "Graphic Designer",
        "Cybersecurity Analyst",
        "Ethical Hacker",
        "Cloud Engineer",
        "DevOps Engineer",
        "Mobile App Developer",
        "Android Developer",
        "iOS Developer",
        "Software Engineer",
        "Game Developer",
        "Blockchain Developer",
        "AR/VR Developer",
        "Database Administrator",
        "Network Engineer",
        "System Administrator",
        "Embedded Systems Engineer",
        "IoT Engineer",
        "QA/Test Engineer",
        "Automation Test Engineer",
        "Digital Marketing Specialist",
        "SEO Specialist",
        "Content Creator",
        "Technical Writer",
        "Product Manager",
        "Project Manager",
        "Robotics Engineer",
        "Computer Vision Engineer",
        "NLP Engineer",
        "Big Data Engineer",
        "Data Engineer"
    ],

    "Required_Skills": [
        "Python, Statistics, Machine Learning, SQL, PowerBI, Tableau, Data Visualization",
        "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, NLP",
        "Python, AI, Deep Learning, NLP, TensorFlow, OpenCV",
        "Excel, SQL, PowerBI, Tableau, Statistics, Python",
        "Communication, Excel, SQL, Data Analysis, Problem Solving",

        "HTML, CSS, JavaScript, React, NodeJS",
        "HTML, CSS, JavaScript, React, Bootstrap, Tailwind",
        "Python, Java, NodeJS, APIs, Database, Django",
        "HTML, CSS, JavaScript, React, NodeJS, MongoDB, SQL",

        "Figma, UX Research, Wireframing, Creativity, Prototyping",
        "Photoshop, Illustrator, Creativity, Branding, UI Design",

        "Networking, Linux, Security, Ethical Hacking, Wireshark",
        "Kali Linux, Penetration Testing, Networking, Cybersecurity",

        "AWS, Azure, Docker, Kubernetes, Linux, Cloud Computing",
        "Docker, Kubernetes, Jenkins, Linux, AWS, CI/CD",

        "Flutter, Firebase, Android, iOS, APIs",
        "Java, Kotlin, Android Studio, Firebase, XML",
        "Swift, iOS, Xcode, UIKit, APIs",

        "Java, Python, C++, OOP, DSA, Problem Solving",
        "Unity, C#, Unreal Engine, Game Design, Animation",

        "Blockchain, Solidity, Ethereum, Smart Contracts, Web3",
        "Unity, Blender, AR, VR, 3D Modeling",

        "SQL, MySQL, Oracle, Database Management",
        "Networking, Cisco, Routing, Switching, Firewall",
        "Linux, Windows Server, Networking, Troubleshooting",

        "C, C++, Microcontrollers, Embedded Systems, Arduino",
        "IoT, Sensors, Raspberry Pi, Arduino, MQTT",

        "Manual Testing, Automation Testing, Selenium, JUnit",
        "Selenium, Python, Java, TestNG, Automation",

        "SEO, SEM, Social Media Marketing, Analytics",
        "SEO, Keyword Research, Google Analytics, Content Marketing",

        "Video Editing, Canva, Social Media, Creativity",
        "Documentation, Communication, Technical Knowledge",

        "Leadership, Agile, Scrum, Communication, Planning",
        "Management, Agile, Risk Management, Team Handling",

        "Python, Robotics, ROS, Automation, AI",
        "Python, OpenCV, Deep Learning, Image Processing",
        "Python, NLP, Transformers, Deep Learning, AI",

        "Hadoop, Spark, Kafka, Big Data, Python",
        "Python, SQL, ETL, Data Warehousing, Spark"
    ],

    "Description": [
        "Analyze data and create predictive models.",
        "Build and optimize machine learning systems.",
        "Develop intelligent AI applications and systems.",
        "Interpret and visualize business data.",
        "Improve business decisions using data insights.",

        "Build websites and responsive web applications.",
        "Design interactive and responsive user interfaces.",
        "Develop server-side applications and APIs.",
        "Handle both frontend and backend development.",

        "Create user-friendly app and website designs.",
        "Design creative graphics and branding materials.",

        "Protect systems and networks from cyber attacks.",
        "Perform penetration testing and vulnerability assessment.",

        "Manage scalable cloud infrastructure.",
        "Automate deployment pipelines and operations.",

        "Build cross-platform mobile applications.",
        "Develop Android applications.",
        "Develop iOS applications.",

        "Design and build software applications.",
        "Create games using modern game engines.",

        "Develop decentralized blockchain applications.",
        "Build immersive AR and VR experiences.",

        "Manage and maintain databases.",
        "Configure and maintain computer networks.",
        "Maintain servers and IT systems.",

        "Design hardware-integrated software systems.",
        "Develop smart IoT-based systems.",

        "Test software for bugs and quality issues.",
        "Automate software testing processes.",

        "Promote brands through digital marketing.",
        "Improve website ranking on search engines.",

        "Create engaging online content and videos.",
        "Write technical guides and documentation.",

        "Manage software products and teams.",
        "Handle projects, planning, and delivery.",

        "Develop intelligent robotic systems.",
        "Build image recognition and vision systems.",
        "Develop AI systems for language understanding.",

        "Process and manage large-scale data systems.",
        "Build data pipelines and data infrastructure."
    ]
}

df = pd.DataFrame(career_data)
# ----------------------------
# TITLE
# ----------------------------
st.title("🎯 AI Career Advisor")
st.write("Find your perfect career match based on your skills")

# ----------------------------
# INPUT
# ----------------------------
user_input = st.text_area("Enter your skills (comma separated)", "Python, Statistics, HTML")

# ----------------------------
# BUTTON
# ----------------------------
if st.button("Find Careers"):

    vectorizer = TfidfVectorizer()
    skill_matrix = vectorizer.fit_transform(df["Required_Skills"])
    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(user_vector, skill_matrix).flatten()

    # ----------------------------
    # Convert to Percentage
    # ----------------------------
    df["Match_Score"] = similarity_scores * 100

    # ----------------------------
    # Remove 0% matches
    # ----------------------------
    df_filtered = df[df["Match_Score"] > 0]

    # ----------------------------
    # Top 5 results
    # ----------------------------
    recommendations = df_filtered.sort_values(
    by="Match_Score",
    ascending=False).head(5)

    recommendations = recommendations.reset_index(drop=True)
    recommendations.index=range(1,len(recommendations)+1)

    st.subheader("🔮 Top Career Matches")

    # Format percentage nicely
    recommendations_display = recommendations.copy()
    recommendations_display["Match_Score"] = recommendations_display["Match_Score"].apply(lambda x: f"{x:.2f}%")

    st.dataframe(
        recommendations_display[["Career", "Match_Score"]],
        use_container_width=True
    )

    # ----------------------------
    # Skill Gap Analysis
    # ----------------------------
    st.subheader("📊 Skill Gap Analysis")

    user_skills = set([s.strip().lower() for s in user_input.split(",")])

    for _, row in recommendations.iterrows():
        career = row["Career"]

        required_skills = set([s.strip().lower() for s in row["Required_Skills"].split(",")])
        missing_skills = required_skills - user_skills

        st.markdown(f"""
        <div style="
            background-color:#1e293b;
            padding:15px;
            border-radius:12px;
            margin-bottom:10px;
            color:white;
        ">
        <h4 style="color:#38bdf8;">{career}</h4>
        <p><b>Match Score:</b> {row["Match_Score"]:.2f}%</p>
        <p><b>Missing Skills:</b> {', '.join(missing_skills) if missing_skills else 'None 🎉'}</p>
        </div>
        """, unsafe_allow_html=True)