<template>
  <div
    class="bubble-button"
    :class="{ 'large': size === 'large', 'medium': size === 'medium', 'small': size === 'small' }"
    :style="{ backgroundColor: color }"
    @click="$emit('click')"
  >
    <div class="bubble-content">
      <slot></slot>
    </div>
  </div>
</template>

<script setup>
defineProps({
  color: {
    type: String,
    default: '#4A6163'
  },
  size: {
    type: String,
    default: 'medium',
    validator: (value) => ['small', 'medium', 'large'].includes(value)
  }
});

defineEmits(['click']);
</script>

<style scoped>
.bubble-button {
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow: hidden;
  width: 100%;
  height: 100%;
}

.bubble-button:hover {
  transform: translateY(-5px);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
}

.bubble-button:active {
  transform: translateY(0);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.bubble-button::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 70%);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.bubble-button:hover::after {
  opacity: 1;
}

.bubble-content {
  text-align: center;
  padding: 1rem;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

/* These sizes are now used as max-width/max-height */
.small {
  max-width: 120px;
  max-height: 120px;
  font-size: 0.9rem;
}

.medium {
  max-width: 180px;
  max-height: 180px;
  font-size: 1.1rem;
}

.large {
  max-width: 240px;
  max-height: 240px;
  font-size: 1.3rem;
}
</style>
