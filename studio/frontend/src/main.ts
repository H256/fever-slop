import { createApp } from "vue";
import { createPinia } from "pinia";
import { createRouter, createWebHistory } from "vue-router";
import App from "./App.vue";
import "./styles.css";
import ProjectsPage from "./pages/ProjectsPage.vue";
import ProjectDashboard from "./pages/ProjectDashboard.vue";
import PipelinePage from "./pages/PipelinePage.vue";
import RenderPlanPage from "./pages/RenderPlanPage.vue";
import ReferencesPage from "./pages/ReferencesPage.vue";
import ArtifactsPage from "./pages/ArtifactsPage.vue";
import RenderQueuePage from "./pages/RenderQueuePage.vue";
import ReviewPage from "./pages/ReviewPage.vue";
import SettingsPage from "./pages/SettingsPage.vue";
import ProjectSettingsPage from "./pages/ProjectSettingsPage.vue";
import FinalVideoPage from "./pages/FinalVideoPage.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: ProjectsPage },
    { path: "/projects/:projectId", component: ProjectDashboard },
    { path: "/projects/:projectId/pipeline", component: PipelinePage },
    { path: "/projects/:projectId/render-plan", component: RenderPlanPage },
    { path: "/projects/:projectId/references", component: ReferencesPage },
    { path: "/projects/:projectId/artifacts", component: ArtifactsPage },
    { path: "/projects/:projectId/settings", component: ProjectSettingsPage },
    { path: "/projects/:projectId/queue", component: RenderQueuePage },
    { path: "/projects/:projectId/review", component: ReviewPage },
    { path: "/projects/:projectId/final-video", component: FinalVideoPage },
    { path: "/settings", component: SettingsPage }
  ]
});

createApp(App).use(createPinia()).use(router).mount("#app");
