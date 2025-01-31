# 所有和ai交互的代码放进utils.py里（utils 通常是 “utilities” 的缩写，意为 “实用工具” 或 “实用函数”）

from langchain.prompts import ChatPromptTemplate   
from langchain_openai import ChatOpenAI
from langchain_community.utilities import WikipediaAPIWrapper # （“wrap” 常见的意思是 “封装”“包装”）该类内部封装了对维基百科 API 的调用，它会用维基百科官方的API进行搜索，并且返回给我们搜索结果的摘要
import os

# 根据主题和时长，规定创造性，获得视频的标题和脚本
def generate_script(subject,video_length,
                    creativity,api_key):
    
    model = ChatOpenAI(api_key=api_key,base_url="https://api.gptsapi.net/v1",
                       temperature=creativity) # 初始化模型

    # 获得视频的标题
    title_template = ChatPromptTemplate.from_messages(
        [
            ("human","请为主题为‘{subject}’的视频起一个吸引人的标题")
        ]
    ) # 定义提示模板
    title = (title_template|model).invoke(
        {
            "subject":subject
        }
    ).content # 调用链的invoke，获得最终结果


    # 调用维基百科的API获得相关信息（先创建 WikipediaAPIWrapper 的实例，然后调用其 run 方法来执行搜索操作）
    search = WikipediaAPIWrapper(lang="zh") #创建WikipediaAPIWrapper实例，指定搜索中文的内容
    search_result = search.run(subject)

    # 获得视频的脚本内容
    script_template = ChatPromptTemplate.from_messages(
        [
            ("human",
            """你是一位短视频频道的博主。根据以下标题和相关信息，为短视频频道写一个视频脚本。
            视频标题：{title}，视频时长：{duration}分钟，生成的脚本的长度尽量遵循视频时长的要求。
            要求开头抓住眼球，中间提供干货内容，结尾有惊喜，脚本格式也请按照【开头、中间，结尾】分隔。
            整体内容的表达方式要尽量轻松有趣，吸引年轻人。
            脚本内容可以结合以下维基百科搜索出的信息，但仅作为参考，只结合相关的即可，对不相关的进行忽略：
            ```{wikipedia_search}```
             """)
        ]
    )
    script = (script_template|model).invoke(
        {
            "title":title,
            "duration":video_length,
            "wikipedia_search":search_result
        }
    ).content

    return search_result,title,script

# print(generate_script("deepseek大模型",0.5,0.7,os.getenv("OPENAI_API_KEY"))) 自定义的函数，此时还不会自动从 .env 文件中读取配置信息，需要用到os.getenv
