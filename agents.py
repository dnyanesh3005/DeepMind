from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from create_tool import get_current_news, scrape_url
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def build_search_agent():
    return create_react_agent(
        model=llm,
        tools=[get_current_news],
    )

def build_read_agent():
    return create_react_agent(
        model=llm,
        tools=[scrape_url],
    )   

#writer chain
writer_prompt=ChatPromptTemplate.from_messages([
    ("system","you are a expert research writer,write clear,structured ans insightfull report"),
    ("human","""write a detailed report on this topic below
    topic:{topic}
    
    Here are the research materials you can use:
    {research}
    
    structure the report as :
    -introduction
    -key findings
    -conclusion
    -sources all the URL found in the research
    be detailed nad professional
    """),

])

writer_chain=writer_prompt | llm | StrOutputParser()

#critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research critic. Your job is to review research reports for accuracy, clarity, depth, and structure. Be honest, constructive, and thorough."),
    ("human", """Review the following research report and provide detailed feedback.

Topic: {topic}

Report:
{report}

Please evaluate the report on the following criteria and give a score out of 10 for each:

1. **Accuracy** Are the facts correct and well-supported?
2. **Depth** Is the topic covered in sufficient detail?
3. **Clarity** Is the writing clear and easy to understand?
4. **Structure** Is the report well-organized (intro, findings, conclusion, sources)?
5. **Source Quality** Are credible URLs/sources cited?

Finally, give:
- **Overall Score**: X/10
- **Strengths**: What the report does well
- **Improvements**: What could be better

Be professional and constructive.
"""),
])
critic_chain=critic_prompt | llm | StrOutputParser()
