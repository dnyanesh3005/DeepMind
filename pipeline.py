from agents import build_search_agent,build_read_agent,writer_chain,critic_chain

def run_research_pipeline(topic: str)-> dict:

    state={}

    print('\n'+'#'*50)
    print("1 search agent is working")
    print('#'*50)

    search_agent=build_search_agent()
    search_result=search_agent.invoke({
        "messages":[("user",f"find the recent information on {topic}")]
    })
    state['search_result']=search_result['messages'][-1].content
    
    print("\n serach agent result",state['search_result'])

    #reader agent
    print("\n"+'#'*50)
    print("2 read agent is working")
    print('#'*50)

    read_agent=build_read_agent()
    read_result=read_agent.invoke({
        "messages":[("user",
        f"based on the following search results `{topic}`"
        f"pick the mosr recent urls and scrap it deeper content"
        f"search result:\n {state['search_result'][:1000]}"

        )]
    })
    state['scraped_content']=read_result['messages'][-1].content

    print("\n scraped content:",state['scraped_content'])

    #step 3 writer chain
    print('\n'+'#'*50)
    print("3 writer agent is working")
    print('#'*50)

    combined_result=(
        f"SEARCH RESULT: \n{state['search_result']}"
        f"\n\n SCRAPED URL DATA\n{state['scraped_content']}"
    )

    state['report']=writer_chain.invoke({
        'topic':topic,
        'research':combined_result
    })
    print("\n\n writer result:",state['report'])

    #critic report
    print('\n'+'#'*50)
    print("4 critic agent is working")
    print('#'*50)

    state['critic_result']=critic_chain.invoke({
        'topic':topic,
        'report':state['report']
    })
    print("\n\n critic result:",state['critic_result'])

    return state
if __name__=="__main__":
    topic=input("enter the topic you want research on:")
    result=run_research_pipeline(topic)
    print("\n Final result:",result) 