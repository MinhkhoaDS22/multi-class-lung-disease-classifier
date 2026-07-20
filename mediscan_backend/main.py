from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import traceback


# ==========================================
# LIFESPAN – Load model khi khởi động
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    from model import mediscan_model
    mediscan_model.load()
    yield
    print("[MediScan] Server đang tắt...")


# ==========================================
# KHỞI TẠO APP
# ==========================================
app = FastAPI(
    title="MediScan AI API",
    description="API phân tích ảnh X-quang ngực bằng EfficientNet-B5",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – Cho phép Flutter Web gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "MediScan AI API đang chạy!",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    from model import mediscan_model
    return {
        "status": "healthy",
        "model_loaded": mediscan_model._loaded,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận ảnh X-quang, trả về xác suất từng loại bệnh.
    - **file**: File ảnh (JPG, PNG, WEBP...)
    """
    # Kiểm tra định dạng file
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không hợp lệ: {file.content_type}. Vui lòng dùng JPG/PNG/WEBP."
        )

    # Kiểm tra kích thước file (tối đa 20MB)
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File quá lớn. Tối đa 20MB.")

    try:
        from model import mediscan_model
        result = mediscan_model.predict(contents)
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            **result
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


# ==========================================
# CHẠY SERVER
# ==========================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
