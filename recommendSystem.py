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

# 全局变量
MODEL_CACHE_FILE = "model_cache.pkl"
embedding_cache = {}

# 模型加载优化
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 定义数据模型
class userData(BaseModel):
    userModel: Dict[str, float]  # 用户模型,键为标签名(str), 值为权重(float)
    videos: Dict[str, List[str]]  # 待推荐视频的标签列表, 键为视频标题(str), 值为标签列表(List[str])
    likeVideos: List[str] = []  # 用户喜欢的视频标题列表
    favoriteVideos: List[str] = []  # 用户收藏的视频标题列表
    recommend_count: int = 5  # 推荐视频数量

# 创建 logger
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("开始加载模型...")
    load_or_create_model()
    logger.info("模型加载完成")
    logger.info("服务已启动")
    yield
    # 关闭时执行
    logger.info("服务关闭")

app = FastAPI(lifespan=lifespan)

@app.post("/recommend/")
async def recommend(data: userData):
    try:
        logger.info("开始处理推荐请求...")
        # 1. 构建用户向量
        # 创建512维的零向量(模型输出维度)
        user_embedding = np.zeros(512)  # distiluse-base-multilingual-cased-v2 输出维度为512
        # 计算所有标签权重总和，用于归一化
        total_weight = sum(data.userModel.values())
        
        # 对每个用户标签进行向量化并加权
        for tag, weight in data.userModel.items():
            # 将标签转换为向量
            tag_embedding = model.encode(tag)
            # 加权累加到用户向量中
            user_embedding += (tag_embedding * (weight / total_weight))
        
        # 2. 构建视频向量
        video_embeddings = {}
        for video_name, tags in data.videos.items():
            # 将视频的所有标签转换为向量并取平均
            video_embedding = np.mean([model.encode(tag) for tag in tags], axis=0)
            video_embeddings[video_name] = video_embedding

        # 3. 标签相似度计算
        tag_similarities = {}
        for video_name, video_embedding in video_embeddings.items():
             # 计算用户向量和视频向量的余弦相似度
            tag_similarity = cosine_similarity([user_embedding], [video_embedding])[0][0]
            tag_similarities[video_name] = tag_similarity
            
        # 4. 标题相似度计算
        title_similarities = {}
        FAVORITE_WEIGHT = 0.7  # 收藏视频权重
        LIKE_WEIGHT = 0.3     # 喜欢视频权重
        for video_name in data.videos.keys():
            # 将当前视频标题转换为向量
            current_title_embedding = model.encode(video_name)
            
            # 如果没有喜欢和收藏的视频，使用标签相似度
            if not data.likeVideos and not data.favoriteVideos:
                title_similarity = max([
                    cosine_similarity([current_title_embedding], [model.encode(tag)])[0][0]
                    for tag in data.userModel.keys()
                ])
            else:
                # 计算与收藏视频的相似度
                favorite_similarities = [
                    cosine_similarity([current_title_embedding], [model.encode(fav_title)])[0][0]
                    for fav_title in data.favoriteVideos
                ] if data.favoriteVideos else [0]
                
                # 计算与喜欢视频的相似度
                like_similarities = [
                    cosine_similarity([current_title_embedding], [model.encode(like_title)])[0][0]
                    for like_title in data.likeVideos
                ] if data.likeVideos else [0]
                
                # 加权计算最终相似度
                title_similarity = (
                    max(favorite_similarities) * FAVORITE_WEIGHT +
                    max(like_similarities) * LIKE_WEIGHT
                )
                
            title_similarities[video_name] = title_similarity
            
        # 5. 混合相似度计算
        TAG_WEIGHT = 0.4    # 标签权重
        TITLE_WEIGHT = 0.6  # 标题权重
        
        final_similarities = {}
        for video_name in data.videos.keys():
            # 综合计算最终相似度
            final_similarities[video_name] = (
                tag_similarities[video_name] * TAG_WEIGHT +
                title_similarities[video_name] * TITLE_WEIGHT
            )

        # 6. 获取推荐结果
        # 按相似度排序并取前(data.recommend_count)个
        recommended_videos = sorted(
            final_similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:data.recommend_count]
            
        logger.info(f"推荐计算完成，推荐{len(recommended_videos)}个视频")
        # 7. 返回推荐结果
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



if __name__ == "__main__":
    import uvicorn
    logger.info("开始启动服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")