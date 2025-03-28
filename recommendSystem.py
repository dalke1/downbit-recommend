import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
import time
from pypinyin import pinyin, Style
import re
from difflib import SequenceMatcher

# 全局变量
MODEL_CACHE_FILE = "model_cache.pkl"
VECTOR_CACHE_FILE = "vector_cache.pkl"
vector_cache = {}

def load_or_create_model():
    global model
    if os.path.exists(MODEL_CACHE_FILE):
        logger.info("从缓存加载模型...")
        with open(MODEL_CACHE_FILE, 'rb') as f:
            model = pickle.load(f)
    else:
        logger.info("首次加载模型...")
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        with open(MODEL_CACHE_FILE, 'wb') as f:
            pickle.dump(model, f)
    return model

def load_vector_cache():
    global vector_cache
    if os.path.exists(VECTOR_CACHE_FILE):
        logger.info("从缓存加载向量...")
        with open(VECTOR_CACHE_FILE, 'rb') as f:
            vector_cache = pickle.load(f)
    else:
        vector_cache = {}
    return vector_cache

def save_vector_cache():
    with open(VECTOR_CACHE_FILE, 'wb') as f:
        pickle.dump(vector_cache, f)

def get_embedding(text: str) -> np.ndarray:
    if text in vector_cache:
        return vector_cache[text]
    embedding = model.encode(text)
    vector_cache[text] = embedding
    return embedding

def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    cached = []
    uncached = []
    for text in texts:
        if text in vector_cache:
            cached.append(text)
        else:
            uncached.append(text)
    
    embeddings = [vector_cache[text] for text in cached]
    if uncached:
        new_embeddings = model.encode(uncached)
        for text, emb in zip(uncached, new_embeddings):
            vector_cache[text] = emb
        embeddings.extend(new_embeddings)
    return embeddings

def convert_to_pinyin_and_lowercase(text: str) -> str:
    """
    将字符串中的汉字转换为拼音，保持英文和其他字符不变，最后转换为小写
    
    Args:
        text: 输入的字符串
        
    Returns:
        转换后的字符串
    """
    # 使用正则表达式分割文本，保留非汉字部分
    parts = re.finditer(r'([\u4e00-\u9fff]+|[^\u4e00-\u9fff]+)', text)
    
    result = []
    for part in parts:
        content = part.group()
        # 检查是否为汉字
        if re.match(r'[\u4e00-\u9fff]+', content):
            # 转换汉字为拼音
            py = pinyin(content, style=Style.NORMAL)
            # 将拼音列表展平并连接
            result.append(''.join([item[0] for item in py]))
        else:
            # 非汉字部分保持不变
            result.append(content)
    
    # 连接所有部分并转换为小写
    return ''.join(result).lower()




# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class userData(BaseModel):
    userModel: Dict[str, float] = {}
    videos: Dict[str, List[str]]
    likeVideos: List[str] = []
    favoriteVideos: List[str] = []
    recommend_count: int = 5
    keyWords: List[str] = []

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("开始加载模型...")
    load_or_create_model()
    load_vector_cache()
    logger.info("模型和向量缓存加载完成")
    yield
    logger.info("保存向量缓存...")
    save_vector_cache()
    logger.info("服务关闭")

app = FastAPI(lifespan=lifespan)

@app.post("/recommend/")
async def recommend(data: userData):
    try:
        start_time = time.perf_counter()
        logger.info("开始处理推荐请求...")

        # 构建视频向量（无论是否有关键词都需要）
        video_embeddings = {}
        for video_name, tags in data.videos.items():
            tag_embeddings = get_embeddings(tags)
            video_embeddings[video_name] = np.mean(tag_embeddings, axis=0)

        final_similarities = {}
        # 判断是否使用关键词推荐
        if data.keyWords:
            logger.info(f"使用关键词推荐模式: {data.keyWords}")
            
            # 获取视频标题向量
            video_titles = list(data.videos.keys())
            title_embeddings = get_embeddings(video_titles)
            
            # 获取关键词向量（不缓存）
            keyword_embeddings = [model.encode(keyword) for keyword in data.keyWords]
            
            # 计算每个视频的相似度
            for video_name, title_embedding, video_embding in zip(video_titles, title_embeddings, video_embeddings.values()):
                # 1. 关键词与标题的相似度
                pinyin_video_name = convert_to_pinyin_and_lowercase(video_name)
                exact_match_bonus = 1.0
                pinyin_match = [];
                for keyword in data.keyWords:
                    pinyin_keyword = convert_to_pinyin_and_lowercase(keyword)
                    pinyin_match.append(SequenceMatcher(None, pinyin_video_name, pinyin_keyword).get_matching_blocks())
                    if len(pinyin_match[-1]) > 1:
                        exact_match_bonus = 2.0
                        break
                title_sims = [
                    cosine_similarity([title_embedding], [keyword_embding])[0][0]
                    for keyword_embding in keyword_embeddings
                ]
                max_title_sim = max(title_sims) * exact_match_bonus
                
                # 2. 关键词与视频标签的相似度
                tag_sims = [
                    cosine_similarity([video_embding], [keyword_embding])[0][0]
                    for keyword_embding in keyword_embeddings
                ]
                max_tag_sim = max(tag_sims)
                
                # 3. 混合计算最终相似度
                KEYWORD_TITLE_WEIGHT = 0.8
                KEYWORD_TAG_WEIGHT = 0.2
                final_similarities[video_name] = (
                    max_title_sim * KEYWORD_TITLE_WEIGHT + 
                    max_tag_sim * KEYWORD_TAG_WEIGHT
                )
        
        else:
            logger.info("使用用户模型推荐模式")
            # 1. 构建用户向量
            user_embedding = np.zeros(512)
            total_weight = sum(data.userModel.values())
            tag_embeddings = get_embeddings(list(data.userModel.keys()))
            
            for embedding, weight in zip(
                tag_embeddings, 
                data.userModel.values()
            ):
                user_embedding += (embedding * (weight / total_weight))

            # 2. 计算标签相似度
            tag_similarities = {
                video_name: cosine_similarity([user_embedding], [video_emb])[0][0]
                for video_name, video_emb in video_embeddings.items()
            }

            # 3. 计算标题相似度
            title_similarities = {}
            FAVORITE_WEIGHT = 0.7
            LIKE_WEIGHT = 0.3

            video_titles = list(data.videos.keys())
            current_title_embeddings = get_embeddings(video_titles)
            fav_embeddings = get_embeddings(data.favoriteVideos) if data.favoriteVideos else []
            like_embeddings = get_embeddings(data.likeVideos) if data.likeVideos else []

            for video_name, current_title_embedding in zip(video_titles, current_title_embeddings):
                if not data.likeVideos and not data.favoriteVideos:
                    title_similarities[video_name] = cosine_similarity([current_title_embedding], [user_embedding])[0][0]
                else:
                    fav_max = max(cosine_similarity([current_title_embedding], fav_embeddings)[0]) if fav_embeddings else 0
                    like_max = max(cosine_similarity([current_title_embedding], like_embeddings)[0]) if like_embeddings else 0
                    title_similarities[video_name] = (fav_max * FAVORITE_WEIGHT + like_max * LIKE_WEIGHT)

            # 4. 混合相似度计算
            TAG_WEIGHT = 0.4
            TITLE_WEIGHT = 0.6
            final_similarities = {
                video_name: (tag_similarities[video_name] * TAG_WEIGHT + 
                           title_similarities[video_name] * TITLE_WEIGHT)
                for video_name in data.videos.keys()
            }

        # 获取排序后的推荐结果
        recommended_videos = sorted(
            final_similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:data.recommend_count]
        
        # 计算处理时间
        process_time = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"推荐计算耗时: {process_time} 毫秒")
        logger.info(f"推荐计算完成，推荐{len(recommended_videos)}个视频")
        logger.info("推荐结果:")
        for video_title, score in recommended_videos:
            logger.info(f"{video_title}: {score}")
        return {
            "recommendations": [
                {
                    "video_title": video_title,
                    "score": round(float(score) * 100, 2)
                } 
                for video_title, score in recommended_videos
            ]
        }
        
    except Exception as e:
        logger.error(f"推荐过程出错: {str(e)}")
        return {"status": "error", "message": str(e)}


# 添加新的请求模型类
class RelatedReqDto(BaseModel):
    videoTitle: str
    tags: List[str]
    videos: Dict[str, List[str]]

# 修改 related 接口实现
@app.post("/related/")
async def related(data: RelatedReqDto):
    try:
        start_time = time.perf_counter()
        logger.info(f"开始计算视频 '{data.videoTitle}' 的相似视频...")

        # 1. 获取目标视频的向量表示
        target_title_embedding = get_embedding(data.videoTitle)
        target_tags_embeddings = get_embeddings(data.tags)
        target_tags_avg_embedding = np.mean(target_tags_embeddings, axis=0)

        # 2. 计算其他视频的相似度
        similarities = {}
        for video_name, tags in data.videos.items():

            # 获取当前视频的标题和标签向量
            current_title_embedding = get_embedding(video_name)
            current_tags_embeddings = get_embeddings(tags)
            current_tags_avg_embedding = np.mean(current_tags_embeddings, axis=0)

            # 计算标题相似度
            title_similarity = cosine_similarity(
                [target_title_embedding], 
                [current_title_embedding]
            )[0][0]

            # 计算标签相似度
            tags_similarity = cosine_similarity(
                [target_tags_avg_embedding], 
                [current_tags_avg_embedding]
            )[0][0]

            # 混合相似度计算 (标题权重0.6，标签权重0.4)
            TITLE_WEIGHT = 0.6
            TAGS_WEIGHT = 0.4
            final_similarity = (
                title_similarity * TITLE_WEIGHT + 
                tags_similarity * TAGS_WEIGHT
            )

            similarities[video_name] = final_similarity

        # 3. 获取排序后的相似视频
        SIMILARITY_THRESHOLD = 0.4  # 40% 相似度阈值
        similar_videos = sorted(
            [
                (video_name, score) 
                for video_name, score in similarities.items() 
                if score >= SIMILARITY_THRESHOLD
            ],
            key=lambda x: x[1],
            reverse=True
        )

        # 计算处理时间
        process_time = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"相似视频计算耗时: {process_time} 毫秒")
        logger.info(f"计算完成，找到{len(similar_videos)}个相似视频")
        logger.info("相似视频列表:")
        for video_title, score in similar_videos:
            logger.info(f"{video_title}: {score}")

        return {
            "recommendations": [
                {
                    "video_title": video_title,
                    "score": round(float(score) * 100, 2)
                } 
                for video_title, score in similar_videos
            ]
        }

    except Exception as e:
        logger.error(f"计算相似视频时出错: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    logger.info("开始启动服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")