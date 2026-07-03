<script setup lang="ts">
import { Image, Video } from "lucide-vue-next";
import {
  displayRenderPlanFieldValue,
  fieldLabel,
  generationTargetsForField,
  optionsForRenderPlanField,
  type RenderPlanField,
  type RenderPlanFormGroup
} from "../composables/renderPlanForm";

defineProps<{
  groups: RenderPlanFormGroup[];
}>();

defineEmits<{
  updateField: [field: RenderPlanField, event: Event];
}>();
</script>

<template>
  <div class="generated-form">
    <fieldset v-for="group in groups" :key="group.key" class="form-block">
      <legend>{{ group.title }}</legend>
      <label v-for="field in group.fields" :key="field.path.join('.')">
        <span class="field-heading">
          <span class="field-title">{{ fieldLabel(field, group) }}</span>
          <span v-if="generationTargetsForField(field).length" class="generation-icons" aria-label="Used in generation">
            <Image v-if="generationTargetsForField(field).includes('image')" :size="15" />
            <Video v-if="generationTargetsForField(field).includes('video')" :size="15" />
          </span>
        </span>
        <span class="field-help">{{ field.help }}</span>
        <select v-if="optionsForRenderPlanField(field).length" :value="displayRenderPlanFieldValue(field)" @change="$emit('updateField', field, $event)">
          <option v-for="option in optionsForRenderPlanField(field)" :key="option" :value="option">{{ option }}</option>
        </select>
        <input
          v-else-if="field.kind === 'boolean'"
          type="checkbox"
          :checked="Boolean(field.value)"
          @change="$emit('updateField', field, $event)"
        />
        <input
          v-else-if="field.kind === 'number'"
          type="number"
          :value="field.value"
          @input="$emit('updateField', field, $event)"
        />
        <textarea
          v-else-if="field.kind === 'longText'"
          class="transcript-area"
          :value="displayRenderPlanFieldValue(field)"
          @input="$emit('updateField', field, $event)"
        />
        <input
          v-else
          type="text"
          :value="displayRenderPlanFieldValue(field)"
          @input="$emit('updateField', field, $event)"
        />
      </label>
    </fieldset>
  </div>
</template>
