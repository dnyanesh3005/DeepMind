from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from create_tool import get_current_news, scrape_url
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ── Agent builders ────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# WRITER CHAIN  —  few-shot prompting to prevent hallucination
# ══════════════════════════════════════════════════════════════════════════════

_writer_examples = [
    {
        "topic": "Recent advances in solid-state batteries",
        "research": (
            "Source: TechCrunch (https://techcrunch.com/2024/03/solid-batteries)\n"
            "QuantumScape reported a 15% energy density improvement in Q1 2024.\n"
            "Toyota plans to commercialise solid-state EVs by 2027-2028.\n"
            "Key challenge: manufacturing at scale remains costly."
        ),
        "report": (
            "## Introduction\n"
            "Solid-state batteries are emerging as the next frontier in energy storage, "
            "replacing liquid electrolytes with solid materials to improve safety and energy density.\n\n"
            "## Key Findings\n"
            "- **QuantumScape** announced a 15% improvement in energy density in Q1 2024 "
            "(TechCrunch, 2024).\n"
            "- **Toyota** targets commercialisation of solid-state EV batteries between 2027 and 2028.\n"
            "- High manufacturing costs remain the primary barrier to mass adoption.\n\n"
            "## Conclusion\n"
            "Solid-state technology shows strong momentum but commercial viability depends on "
            "solving production-cost challenges in the next 2–3 years.\n\n"
            "## Sources\n"
            "- https://techcrunch.com/2024/03/solid-batteries"
        ),
    },
    {
        "topic": "Impact of generative AI on software jobs",
        "research": (
            "Source: Bloomberg (https://bloomberg.com/ai-jobs-2024)\n"
            "Survey of 1,200 tech firms: 38% plan to hire fewer junior developers due to AI tools.\n"
            "GitHub Copilot users report 55% faster code completion on average.\n"
            "New roles emerging: AI prompt engineers, AI QA testers.\n"
            "Source: MIT study (https://mit.edu/ai-labor-2024) — AI augments rather than replaces "
            "senior engineers."
        ),
        "report": (
            "## Introduction\n"
            "Generative AI is reshaping the software industry, prompting debate about job displacement "
            "versus role transformation.\n\n"
            "## Key Findings\n"
            "- **38%** of tech firms plan to reduce junior developer hiring as AI coding tools improve "
            "(Bloomberg, 2024).\n"
            "- GitHub Copilot users complete code tasks **55% faster** on average.\n"
            "- New specialised roles are emerging, including AI prompt engineers and AI QA testers.\n"
            "- An MIT study (2024) concludes that AI primarily **augments** senior engineers "
            "rather than replacing them.\n\n"
            "## Conclusion\n"
            "Generative AI is transforming—not eliminating—software roles. Junior positions face "
            "the most pressure, while experienced engineers gain productivity leverage.\n\n"
            "## Sources\n"
            "- https://bloomberg.com/ai-jobs-2024\n"
            "- https://mit.edu/ai-labor-2024"
        ),
    },
]

_writer_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Topic: {topic}\n\nResearch:\n{research}"),
    ("ai", "{report}"),
])

_writer_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=_writer_example_prompt,
    examples=_writer_examples,
)

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert research writer. Your reports must be:\n"
            "1. GROUNDED ONLY in the provided research — never invent facts, statistics, or sources.\n"
            "2. Clearly structured with Introduction, Key Findings, Conclusion, and Sources sections.\n"
            "3. Professional, detailed, and well-cited.\n"
            "If the research does not mention a fact, do NOT include it."
        ),
    ),
    _writer_few_shot,
    (
        "human",
        "Topic: {topic}\n\nResearch:\n{research}\n\n"
        "Write a detailed, grounded research report using ONLY the facts above.",
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ══════════════════════════════════════════════════════════════════════════════
# CRITIC CHAIN  —  few-shot prompting for consistent, structured feedback
# ══════════════════════════════════════════════════════════════════════════════

_critic_examples = [
    {
        "topic": "Solid-state batteries",
        "report": (
            "## Introduction\nSolid-state batteries are the future...\n"
            "## Key Findings\n- QuantumScape improved energy density by 15%.\n"
            "## Conclusion\nPromising technology.\n## Sources\n- https://techcrunch.com/2024/03/solid-batteries"
        ),
        "feedback": (
            "### Scores\n"
            "| Criterion | Score |\n|---|---|\n"
            "| Accuracy | 8/10 |\n"
            "| Depth | 5/10 |\n"
            "| Clarity | 7/10 |\n"
            "| Structure | 8/10 |\n"
            "| Source Quality | 6/10 |\n\n"
            "**Overall Score: 6.8/10**\n\n"
            "**Strengths:**\n"
            "- Accurately cites QuantumScape's energy density figure with a credible source.\n"
            "- Clear section structure with all required headings.\n\n"
            "**Improvements:**\n"
            "- Key Findings section is too brief — only one data point is discussed.\n"
            "- Conclusion lacks specificity; 'promising technology' is vague.\n"
            "- Only one source cited; the research contained more that could strengthen credibility."
        ),
    },
]

_critic_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "Topic: {topic}\n\nReport:\n{report}"),
    ("ai", "{feedback}"),
])

_critic_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=_critic_example_prompt,
    examples=_critic_examples,
)

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert research critic. Evaluate reports strictly on what is written.\n"
            "NEVER assume facts exist if they are not in the report.\n"
            "Score each criterion out of 10 and give clear, actionable feedback."
        ),
    ),
    _critic_few_shot,
    (
        "human",
        "Topic: {topic}\n\nReport:\n{report}\n\n"
        "Evaluate this report across: Accuracy, Depth, Clarity, Structure, Source Quality.\n"
        "Provide scores in a markdown table, an Overall Score, Strengths, and Improvements."
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
