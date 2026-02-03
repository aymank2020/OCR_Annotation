# 🚀 Deploy Atlas Annotation to Render.com (FREE)

## Quick Start - Deploy in 3 Minutes ⚡

### What You'll Get:
- ✅ **Free web hosting** (512 MB RAM, 0.1 CPU)
- ✅ **HTTPS enabled** automatically
- ✅ **Live URL** (like https://atlas-annotation.onrender.com)
- ✅ **Auto-deploys** on GitHub push
- ✅ **Sleeps after 15 min inactivity**, wakes on request (free tier)

---

## Step 1: Push Your Updated Files to GitHub

Your project already has:
- ✅ `app.py` - Flask web server
- ✅ `web_interface.html` - User interface
- ✅ `requirements.txt` - Dependencies (updated)
- ✅ `Procfile` - Render deployment config
- ✅ `runtime.txt` - Python 3.10.14

Just commit and push:

```bash
cd /teamspace/studios/this_studio/.openclaw/workspace/OCR_Annotation/
git add .
git commit -m "Add Render deployment files"
git push origin main
```

---

## Step 2: Create Render Account (FREE)

1. Go to: https://render.com/
2. Click **"Sign Up"**
3. Use **GitHub** to sign up (easiest)
4. Verify email if required

---

## Step 3: Deploy Your App

### Option A: One-Click from GitHub (Fastest)

1. In Render dashboard:
   - Click **"New +"** → **"Web Service"**
   - Select your **OCR_Annotation** repository
   - Click **"Connect"**

2. Configure the service:
   - **Name:** `atlas-annotation`
   - **Region:** `Oregon (us-west)` (or closest to you)
   - **Branch:** `main`
   - **Root Directory:** `./` (leave default)
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --timeout 120 --workers 1 --bind 0.0.0.0:$PORT`

3. **Important Settings:**
   - **Instance Type:** `Free`
   - **Environment Variables:**
     - None needed (defaults work)

4. Click **"Create Web Service"**

5. **Wait 2-5 minutes** for deployment:
   - Builds → Installs dependencies → Starts server
   - Watch the logs in Render dashboard

6. **Done!** 🎉
   - Click your live URL
   - Access your Atlas Annotation web interface!

---

### Option B: using render.yaml (Automatic)

If you pushed `render.yaml`, Render auto-detects it:

1. Push to GitHub as above
2. In Render dashboard → **"New +"** → **"Web Service"**
3. Select repo → **"Connect"**
4. Use the **render.yaml configuration** (click "Existing Blueprint")
5. **"Create Web Service"**

---

## Step 4: Access Your Live App

Once deployed:
- **Web Interface:** `https://atlas-annotation.onrender.com`
- **API Status:** `https://atlas-annotation.onrender.com/api/status`
- **API Upload:** `POST https://atlas-annotation.onrender.com/api/annotate`

---

## ⚠️ Free Tier Limitations

| Feature | Limitation |
|---------|------------|
| **RAM** | 512 MB |
| **CPU** | 0.1 vCPU (will be slow for large videos) |
| **Sleep** | After 15 min inactivity |
| **Wake time** | ~30 seconds on first request |
| **Storage** | Temporary (files deleted on rebuild) |

**Tips:**
- ❌ Don't upload huge videos (>100 MB) - will be slow
- ✅ Use for annotation validation and small videos
- ✅ Great for testing and demos

---

## 💡 For Production (Paid)

If you need better performance:
- **Starter ($7/mo):** 512 MB RAM, 0.5 vCPU
- **Standard ($25/mo):** 2 GB RAM, 1 vCPU
- **Pro Plus (~$100/mo):** 8 GB RAM, 4 vCPU

Upgrade in Render dashboard anytime.

---

## 🔧 Troubleshooting

### Build Fails
**Error:** `Module not found`
- Check `requirements.txt` has all dependencies
- Rebuild: Render dashboard → "Manual Deploy"

### App Crashes
**Error:** `Out of memory`
- Free tier has 512 MB limit
- Reduce video size or upgrade plan

### 502 Bad Gateway
- Check logs in Render dashboard
- Model might not be initialized
- Try manual deploy again

### Wake Time Too Long
- Free tier sleeps after inactivity
- First request takes ~30 seconds
- Consider paid tier for always-on

---

## 🎯 What You Can Do on the Live App

1. **Web Interface**
   - Upload videos
   - View annotations
   - Download results (JSON, Atlas, CSV)

2. **API Endpoints**
   - Status check
   - Video upload
   - Results retrieval
   - Batch processing

3. **Annotation Validation**
   - Test your NEW_GUIDELINES rules
   - Validate labels
   - Check compliance

---

## 📞 Need Help?

- **Render Docs:** https://render.com/docs
- **Render Status:** https://status.render.com

---

## 🚀 Next Steps

After deploying:

1. ✅ Test live version with small video
2. ✅ Share URL with others
3. ✅ Integrate API into other tools
4. ⬆️ Upgrade plan if needed for performance

---

**Happy annotating!** 🎬