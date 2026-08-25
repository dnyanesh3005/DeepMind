import streamlit as st
from agents import build_search_agent, build_read_agent, writer_chain, critic_chain

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="centered")

st.title("🔎 Multi-Agent Research Assistant")
st.caption("Search → Read/Scrape → Write → Critique, powered by your LangChain agent pipeline.")

topic = st.text_input("Enter a topic to research", placeholder="e.g. Latest developments in solid-state batteries")
run_button = st.button("Run Research", type="primary", disabled=not topic.strip())

if run_button:
    state = {}

    try:
        with st.status("Running research pipeline...", expanded=True) as status:

            # Step 1: Search agent
            status.update(label="🔍 Search agent is working...")
            search_agent = build_search_agent()
            search_result = search_agent.invoke({
                "messages": [("user", f"find the recent information on {topic}")]
            })
            state["search_result"] = search_result["messages"][-1].content
            status.write("✅ Search agent finished")

            # Step 2: Read agent
            status.update(label="📖 Read agent is scraping content...")
            read_agent = build_read_agent()
            read_result = read_agent.invoke({
                "messages": [("user",
                    f"based on the following search results `{topic}`"
                    f"pick the mosr recent urls and scrap it deeper content"
                    f"search result:\n {state['search_result'][:1000]}"
                )]
            })
            state["scraped_content"] = read_result["messages"][-1].content
            status.write("✅ Read agent finished")

            # Step 3: Writer chain
            status.update(label="✍️ Writer agent is drafting the report...")
            combined_result = (
                f"SEARCH RESULT: \n{state['search_result']}"
                f"\n\n SCRAPED URL DATA\n{state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": combined_result
            })
            status.write("✅ Writer agent finished")

            # Step 4: Critic chain
            status.update(label="🧐 Critic agent is reviewing the report...")
            state["critic_result"] = critic_chain.invoke({
                "topic": topic,
                "report": state["report"]
            })
            status.write("✅ Critic agent finished")

            status.update(label="Pipeline complete!", state="complete", expanded=False)

        # ---- Final output ----
        st.subheader("📄 Final Report")
        st.markdown(state["report"])

        st.subheader("🧐 Critic Feedback")
        st.markdown(state["critic_result"])

        # ---- Intermediate steps, hidden by default ----
        with st.expander("Show intermediate agent outputs"):
            st.markdown("**Search Result**")
            st.write(state["search_result"])
            st.markdown("**Scraped Content**")
            st.write(state["scraped_content"])

        st.session_state["last_result"] = state

    except Exception as e:
        st.error(f"Pipeline failed: {e}")

elif "last_result" in st.session_state:
    state = st.session_state["last_result"]
    st.subheader("📄 Final Report")
    st.markdown(state["report"])
    st.subheader("🧐 Critic Feedback")
    st.markdown(state["critic_result"])
    with st.expander("Show intermediate agent outputs"):
        st.markdown("**Search Result**")
        st.write(state["search_result"])
        st.markdown("**Scraped Content**")
        st.write(state["scraped_content"])