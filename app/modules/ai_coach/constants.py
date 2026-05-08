from typing import Dict, Any

TRAINING_SCOPE = {
    "valid": [
        "การเรียนรู้",
        "การพัฒนาทักษะ",
        "การทำงาน",
        "ภาวะผู้นำ",
        "performance",
        "career growth",
        "communication",
        "mindset เพื่อการพัฒนา",
        "competency",
        "teamwork",
        "การบริหารจัดการ",
        "การตัดสินใจ",
        "การแก้ปัญหา",
    ],
    "invalid": [
        "สัตว์เลี้ยงทั่วไป",
        "งานอดิเรกทั่วไป",
        "สุขภาพทั่วไป",
        "ความสัมพันธ์ส่วนตัวทั่วไป",
        "เรื่องส่วนตัวที่ไม่เกี่ยวกับการเรียนรู้หรือการพัฒนา",
    ],
}

PHASE1_RULES: Dict[int, Dict[str, Any]] = {
    1: {
        "key": "topic",
        "question": "อยากปรึกษาโค้ชเรื่องอะไร?",
        "goal": "ระบุหัวข้อที่เกี่ยวข้องกับการเรียนรู้ การพัฒนาทักษะ การทำงาน การเติบโต หรือ competency",
        "required": ["specific_topic"],
        "answer_type": "topic",
        "allow_scope_redirect": True,
    },
    2: {
        "key": "importance",
        "question": "เรื่องนี้มีความสำคัญกับคุณอย่างไร?",
        "goal": "เข้าใจเหตุผลว่าทำไมหัวข้อนี้จึงสำคัญต่อการพัฒนาหรือการทำงานของผู้ใช้",
        "required": ["reason"],
        "answer_type": "reason",
    },
    3: {
        "key": "current_feeling",
        "question": "ตอนนี้คุณมีความคิดหรือความรู้สึกกับเรื่องนี้อย่างไร?",
        "goal": "เข้าใจความคิด ความรู้สึก หรือท่าทีปัจจุบันของผู้ใช้ต่อหัวข้อนี้",
        "required": ["emotion_or_thought"],
        "answer_type": "emotion",
    },
}

PHASE2_RULES: Dict[int, Dict[str, Any]] = {
    1: {
        "key": "goal_main",
        "question": "เป้าหมายที่คุณต้องการคืออะไร?",
        "goal": "ทำให้เป้าหมายหลักชัดเจนและสัมพันธ์กับหัวข้อที่เลือก",
        "required": ["clear_goal"],
        "answer_type": "goal",
    },
    2: {
        "key": "expected_outcome",
        "question": "ถ้าการโค้ชครั้งนี้เป็นประโยชน์กับคุณ ผลลัพธ์ที่อยากเห็นคืออะไร?",
        "goal": "ระบุผลลัพธ์ที่จับต้องได้ เช่น ความชัดเจน แผนงาน การตัดสินใจ หรือพฤติกรรมใหม่",
        "required": ["expected_result"],
        "answer_type": "outcome",
    },
    3: {
        "key": "goal_importance",
        "question": "เป้าหมายนี้สำคัญกับคุณเพราะอะไร?",
        "goal": "เข้าใจแรงจูงใจและคุณค่าที่อยู่เบื้องหลังเป้าหมาย",
        "required": ["goal_importance_reason"],
        "answer_type": "importance",
    },
}

PHASE3_RULES: Dict[int, Dict[str, Any]] = {
    1: {
        "key": "current_score",
        "question": "ถ้าเป้าหมายนี้คือ 10 คะแนน ตอนนี้คุณให้ตัวเองอยู่ที่กี่คะแนน?",
        "goal": "ประเมินสถานะปัจจุบันเทียบกับเป้าหมาย",
        "required": ["current_score"],
        "answer_type": "score",
    },
    2: {
        "key": "current_reality",
        "question": "ตอนนี้สถานการณ์จริงเป็นอย่างไรบ้าง?",
        "goal": "เข้าใจบริบท ปัจจัย และสภาพจริงที่เกี่ยวข้องกับเป้าหมาย",
        "required": ["current_reality"],
        "answer_type": "reality",
    },
    3: {
        "key": "blocker",
        "question": "อะไรคือสิ่งที่ยังทำให้คุณไปไม่ถึงเป้าหมายนี้?",
        "goal": "ระบุอุปสรรค ความกลัว ข้อจำกัด หรือช่องว่างที่สำคัญ",
        "required": ["blocker"],
        "answer_type": "blocker",
    },
}

PHASE4_RULES: Dict[int, Dict[str, Any]] = {
    1: {
        "key": "possible_options",
        "question": "มีวิธีหรือทางเลือกอะไรบ้างที่อาจช่วยให้คุณขยับเข้าใกล้เป้าหมายนี้?",
        "goal": "เปิดทางเลือกที่เป็นไปได้หลายแบบก่อนเลือกวิธีที่เหมาะสม",
        "required": ["options"],
        "answer_type": "options",
    },
    2: {
        "key": "best_option",
        "question": "จากทางเลือกเหล่านั้น วิธีไหนเหมาะกับคุณที่สุดในตอนนี้?",
        "goal": "เลือกแนวทางที่เหมาะสมและเป็นไปได้จริง",
        "required": ["selected_option"],
        "answer_type": "choice",
    },
}

PHASE5_RULES: Dict[int, Dict[str, Any]] = {
    1: {
        "key": "first_action",
        "question": "คุณจะเริ่มลงมือทำเรื่องแรกคืออะไร?",
        "goal": "ระบุ action แรกที่ชัดเจนและทำได้จริง",
        "required": ["first_action"],
        "answer_type": "action",
    },
    2: {
        "key": "commitment",
        "question": "คุณตั้งใจจะลงมือทำเมื่อไร และจะทำอย่างไรให้ตัวเองทำต่อเนื่องได้?",
        "goal": "สร้าง commitment เรื่องเวลา วิธีติดตาม และความต่อเนื่อง",
        "required": ["when", "commitment_method"],
        "answer_type": "commitment",
    },
    3: {
        "key": "confidence",
        "question": "ตอนนี้คุณมั่นใจในแผนนี้มากแค่ไหน?",
        "goal": "ประเมินความมั่นใจและดูว่าต้องปรับแผนให้ทำได้จริงขึ้นหรือไม่",
        "required": ["confidence_level"],
        "answer_type": "confidence",
    },
}

PHASES: Dict[int, Dict[str, Any]] = {
    1: {"name": "Topic", "title": "หัวข้อที่ต้องการโค้ช", "rules": PHASE1_RULES},
    2: {"name": "Goal", "title": "เป้าหมาย", "rules": PHASE2_RULES},
    3: {"name": "Reality", "title": "สถานการณ์ปัจจุบัน", "rules": PHASE3_RULES},
    4: {"name": "Options", "title": "ทางเลือกและวิธีการ", "rules": PHASE4_RULES},
    5: {"name": "Will", "title": "การตัดสินใจลงมือทำ", "rules": PHASE5_RULES},
}