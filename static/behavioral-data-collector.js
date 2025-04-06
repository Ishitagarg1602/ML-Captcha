// behavioral-data-collector.js
class BehaviorCollector {
    constructor() {
        this.data = {
            mouseEvents: [],
            keyEvents: [],
            clickEvents: [],
            scrollEvents: [],
            inputEvents: [],
            sessionInfo: { startTime: Date.now() }
        };
        this.setupListeners();
    }

    setupListeners() {
        // Mouse movement tracking
        document.addEventListener('mousemove', (e) => {
            this.data.mouseEvents.push({
                x: e.clientX,
                y: e.clientY,
                timestamp: Date.now()
            });
        });

        // Keyboard events
        document.addEventListener('keydown', (e) => {
            this.data.keyEvents.push({
                key: e.key,
                eventType: 'keydown',
                timestamp: Date.now()
            });
        });
        document.addEventListener('keyup', (e) => {
            this.data.keyEvents.push({
                key: e.key,
                eventType: 'keyup',
                timestamp: Date.now()
            });
        });

        // Input events
        document.addEventListener('input', (e) => {
            this.data.inputEvents.push({
                value: e.target.value,
                timestamp: Date.now(),
                elementId: e.target.id
            });
        });

        // Click events
        document.addEventListener('click', (e) => {
            this.data.clickEvents.push({
                x: e.clientX,
                y: e.clientY,
                timestamp: Date.now()
            });
        });

        // Scroll events
        document.addEventListener('scroll', (e) => {
            this.data.scrollEvents.push({
                scrollY: window.scrollY,
                timestamp: Date.now()
            });
        });

        // Focus and blur events
        document.addEventListener('focus', (e) => {
            this.data.inputEvents.push({
                eventType: 'focus',
                elementId: e.target.id,
                timestamp: Date.now()
            });
        }, true);

        document.addEventListener('blur', (e) => {
            this.data.inputEvents.push({
                eventType: 'blur',
                elementId: e.target.id,
                timestamp: Date.now()
            });
        }, true);
    }

    exportData() {
        this.data.sessionInfo.endTime = Date.now();
        this.data.sessionInfo.duration = this.data.sessionInfo.endTime - this.data.sessionInfo.startTime;
        return this.data;
    }
}