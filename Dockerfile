# استخدم Python
FROM python:3.10

# تحديد فولدر الشغل
WORKDIR /app

# نسخ الملفات
COPY . /app

# تثبيت المكتبات
RUN pip install --no-cache-dir -r requirements.txt

# تشغيل السيرفر
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]