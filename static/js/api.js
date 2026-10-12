/**
 * FiberGuard Frontend - API Client
 * Manages communication with FastAPI backend endpoints with graceful fallback
 * to client-side physics/ML if backend server is unreachable.
 */

const API = {
  isBackendOnline: false,
  activeModel: 'Random Forest',
  basePath: '',

  async checkHealth() {
    try {
      const res = await fetch('/api/health', { signal: AbortSignal.timeout(1500) });
      if (res.ok) {
        const data = await res.json();
        this.isBackendOnline = true;
        this.activeModel = data.active_model;
        return data;
      }
    } catch (e) {
      // Backend not reached, running in standalone mode
    }
    this.isBackendOnline = false;
    return { status: 'standalone', active_model: 'Client-side k-NN (Offline Fallback)' };
  },

  async simulate(params) {
    if (this.isBackendOnline) {
      try {
        const res = await fetch('/api/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        });
        if (res.ok) return await res.json();
      } catch (e) {
        console.warn('[API] Backend simulate failed, falling back to client physics', e);
      }
    }
    return ClientPhysics.simulate(params);
  },

  async predict(payload) {
    if (this.isBackendOnline) {
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) return await res.json();
      } catch (e) {
        console.warn('[API] Backend predict failed, falling back to client ML', e);
      }
    }
    return ClientPhysics.predict(payload);
  },

  async getModels() {
    if (this.isBackendOnline) {
      try {
        const res = await fetch('/api/models');
        if (res.ok) return await res.json();
      } catch (e) {
        console.warn('[API] Backend getModels failed', e);
      }
    }
    return ClientPhysics.getMockBenchmarks();
  },

  async selectModel(modelName) {
    if (this.isBackendOnline) {
      try {
        const res = await fetch('/api/models/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: modelName })
        });
        if (res.ok) {
          const data = await res.json();
          this.activeModel = data.active_model;
          return data;
        }
      } catch (e) {
        console.warn('[API] selectModel failed', e);
      }
    }
    this.activeModel = modelName;
    return { status: 'success', active_model: modelName };
  },

  async trainModel(modelName, nTrain) {
    if (this.isBackendOnline) {
      const res = await fetch('/api/models/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName, n_train: nTrain })
      });
      if (res.ok) return await res.json();
    }
    throw new Error('Retraining requires active FastAPI backend');
  },

  async getDatasetSample() {
    if (this.isBackendOnline) {
      try {
        const res = await fetch('/api/dataset/sample');
        if (res.ok) return await res.json();
      } catch (e) {
        console.warn('[API] getDatasetSample failed', e);
      }
    }
    return ClientPhysics.generateSampleRecords(50);
  },

  async diagnoseCustom(customData) {
    if (this.isBackendOnline) {
      try {
        const res = await fetch('/api/diagnose-custom', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(customData)
        });
        if (res.ok) return await res.json();
      } catch (e) {
        console.warn('[API] diagnoseCustom failed', e);
      }
    }
    return ClientPhysics.diagnoseCustom(customData);
  }
};
