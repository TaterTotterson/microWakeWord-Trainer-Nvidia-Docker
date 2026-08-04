//#region node_modules/@vue/shared/dist/shared.esm-bundler.js
// @__NO_SIDE_EFFECTS__
function e(e) {
	let t = /* @__PURE__ */ Object.create(null);
	for (let n of e.split(",")) t[n] = 1;
	return (e) => e in t;
}
var t = {}, n = [], r = () => {}, i = () => !1, a = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && (e.charCodeAt(2) > 122 || e.charCodeAt(2) < 97), o = (e) => e.startsWith("onUpdate:"), s = Object.assign, c = (e, t) => {
	let n = e.indexOf(t);
	n > -1 && e.splice(n, 1);
}, l = Object.prototype.hasOwnProperty, u = (e, t) => l.call(e, t), d = Array.isArray, f = (e) => x(e) === "[object Map]", p = (e) => x(e) === "[object Set]", m = (e) => x(e) === "[object Date]", h = (e) => typeof e == "function", g = (e) => typeof e == "string", _ = (e) => typeof e == "symbol", v = (e) => typeof e == "object" && !!e, y = (e) => (v(e) || h(e)) && h(e.then) && h(e.catch), b = Object.prototype.toString, x = (e) => b.call(e), S = (e) => x(e).slice(8, -1), C = (e) => x(e) === "[object Object]", w = (e) => g(e) && e !== "NaN" && e[0] !== "-" && "" + parseInt(e, 10) === e, ee = /* @__PURE__ */ e(",key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted"), te = (e) => {
	let t = /* @__PURE__ */ Object.create(null);
	return ((n) => t[n] || (t[n] = e(n)));
}, ne = /-\w/g, T = te((e) => e.replace(ne, (e) => e.slice(1).toUpperCase())), re = /\B([A-Z])/g, E = te((e) => e.replace(re, "-$1").toLowerCase()), ie = te((e) => e.charAt(0).toUpperCase() + e.slice(1)), ae = te((e) => e ? `on${ie(e)}` : ""), oe = (e, t) => !Object.is(e, t), se = (e, ...t) => {
	for (let n = 0; n < e.length; n++) e[n](...t);
}, D = (e, t, n, r = !1) => {
	Object.defineProperty(e, t, {
		configurable: !0,
		enumerable: !1,
		writable: r,
		value: n
	});
}, ce = (e) => {
	let t = parseFloat(e);
	return isNaN(t) ? e : t;
}, le = (e) => {
	let t = g(e) ? Number(e) : NaN;
	return isNaN(t) ? e : t;
}, ue, de = () => ue ||= typeof globalThis < "u" ? globalThis : typeof self < "u" ? self : typeof window < "u" ? window : typeof global < "u" ? global : {};
function fe(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) {
			let r = e[n], i = g(r) ? ge(r) : fe(r);
			if (i) for (let e in i) t[e] = i[e];
		}
		return t;
	}
	if (g(e) || v(e)) return e;
}
var pe = /;(?![^(]*\))/g, me = /:([^]+)/, he = /\/\*[^]*?\*\//g;
function ge(e) {
	let t = {};
	return e.replace(he, "").split(pe).forEach((e) => {
		if (e) {
			let n = e.split(me);
			n.length > 1 && (t[n[0].trim()] = n[1].trim());
		}
	}), t;
}
function O(e) {
	let t = "";
	if (g(e)) t = e;
	else if (d(e)) for (let n = 0; n < e.length; n++) {
		let r = O(e[n]);
		r && (t += r + " ");
	}
	else if (v(e)) for (let n in e) e[n] && (t += n + " ");
	return t.trim();
}
var _e = "itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly", ve = /* @__PURE__ */ e(_e);
_e + "";
function ye(e) {
	return !!e || e === "";
}
function be(e, t) {
	if (e.length !== t.length) return !1;
	let n = !0;
	for (let r = 0; n && r < e.length; r++) n = xe(e[r], t[r]);
	return n;
}
function xe(e, t) {
	if (e === t) return !0;
	let n = m(e), r = m(t);
	if (n || r) return n && r ? e.getTime() === t.getTime() : !1;
	if (n = _(e), r = _(t), n || r) return e === t;
	if (n = d(e), r = d(t), n || r) return n && r ? be(e, t) : !1;
	if (n = v(e), r = v(t), n || r) {
		if (!n || !r || Object.keys(e).length !== Object.keys(t).length) return !1;
		for (let n in e) {
			let r = e.hasOwnProperty(n), i = t.hasOwnProperty(n);
			if (r && !i || !r && i || !xe(e[n], t[n])) return !1;
		}
	}
	return String(e) === String(t);
}
function Se(e, t) {
	return e.findIndex((e) => xe(e, t));
}
var Ce = (e) => !!(e && e.__v_isRef === !0), k = (e) => g(e) ? e : e == null ? "" : d(e) || v(e) && (e.toString === b || !h(e.toString)) ? Ce(e) ? k(e.value) : JSON.stringify(e, we, 2) : String(e), we = (e, t) => Ce(t) ? we(e, t.value) : f(t) ? { [`Map(${t.size})`]: [...t.entries()].reduce((e, [t, n], r) => (e[Te(t, r) + " =>"] = n, e), {}) } : p(t) ? { [`Set(${t.size})`]: [...t.values()].map((e) => Te(e)) } : _(t) ? Te(t) : v(t) && !d(t) && !C(t) ? String(t) : t, Te = (e, t = "") => _(e) ? `Symbol(${e.description ?? t})` : e, A, Ee = class {
	constructor(e = !1) {
		this.detached = e, this._active = !0, this._on = 0, this.effects = [], this.cleanups = [], this._isPaused = !1, this._warnOnRun = !0, this.__v_skip = !0, !e && A && (A.active ? (this.parent = A, this.index = (A.scopes || (A.scopes = [])).push(this) - 1) : (this._active = !1, this._warnOnRun = !1));
	}
	get active() {
		return this._active;
	}
	pause() {
		if (this._active) {
			this._isPaused = !0;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].pause();
			}
			for (e = 0, t = this.effects.length; e < t; e++) this.effects[e].pause();
		}
	}
	resume() {
		if (this._active && this._isPaused) {
			this._isPaused = !1;
			let e, t;
			if (this.scopes) {
				let n = this.scopes.slice();
				for (e = 0, t = n.length; e < t; e++) n[e].resume();
			}
			let n = this.effects.slice();
			for (e = 0, t = n.length; e < t; e++) n[e].resume();
		}
	}
	run(e) {
		if (this._active) {
			let t = A;
			try {
				return A = this, e();
			} finally {
				A = t;
			}
		}
	}
	on() {
		++this._on === 1 && (this.prevScope = A, A = this);
	}
	off() {
		if (this._on > 0 && --this._on === 0) {
			if (A === this) A = this.prevScope;
			else {
				let e = A;
				for (; e;) {
					if (e.prevScope === this) {
						e.prevScope = this.prevScope;
						break;
					}
					e = e.prevScope;
				}
			}
			this.prevScope = void 0;
		}
	}
	stop(e) {
		if (this._active) {
			this._active = !1;
			let t, n;
			for (t = 0, n = this.effects.length; t < n; t++) this.effects[t].stop();
			for (this.effects.length = 0, t = 0, n = this.cleanups.length; t < n; t++) this.cleanups[t]();
			if (this.cleanups.length = 0, this.scopes) {
				let e = this.scopes.slice();
				for (t = 0, n = e.length; t < n; t++) e[t].stop(!0);
				this.scopes.length = 0;
			}
			if (!this.detached && this.parent && !e) {
				let e = this.parent.scopes.pop();
				e && e !== this && (this.parent.scopes[this.index] = e, e.index = this.index);
			}
			this.parent = void 0;
		}
	}
};
function De() {
	return A;
}
var j, Oe = /* @__PURE__ */ new WeakSet(), ke = class {
	constructor(e) {
		this.fn = e, this.deps = void 0, this.depsTail = void 0, this.flags = 5, this.next = void 0, this.cleanup = void 0, this.scheduler = void 0, A && (A.active ? A.effects.push(this) : this.flags &= -2);
	}
	pause() {
		this.flags |= 64;
	}
	resume() {
		this.flags & 64 && (this.flags &= -65, Oe.has(this) && (Oe.delete(this), this.trigger()));
	}
	notify() {
		this.flags & 2 && !(this.flags & 32) || this.flags & 8 || Ne(this);
	}
	run() {
		if (!(this.flags & 1)) return this.fn();
		this.flags |= 2, Ke(this), Ie(this);
		let e = j, t = He;
		j = this, He = !0;
		try {
			return this.fn();
		} finally {
			Le(this), j = e, He = t, this.flags &= -3;
		}
	}
	stop() {
		if (this.flags & 1) {
			for (let e = this.deps; e; e = e.nextDep) Be(e);
			this.deps = this.depsTail = void 0, Ke(this), this.onStop && this.onStop(), this.flags &= -2;
		}
	}
	trigger() {
		this.flags & 64 ? Oe.add(this) : this.scheduler ? this.scheduler() : this.runIfDirty();
	}
	runIfDirty() {
		Re(this) && this.run();
	}
	get dirty() {
		return Re(this);
	}
}, Ae = 0, je, Me;
function Ne(e, t = !1) {
	if (e.flags |= 8, t) {
		e.next = Me, Me = e;
		return;
	}
	e.next = je, je = e;
}
function Pe() {
	Ae++;
}
function Fe() {
	if (--Ae > 0) return;
	if (Me) {
		let e = Me;
		for (Me = void 0; e;) {
			let t = e.next;
			e.next = void 0, e.flags &= -9, e = t;
		}
	}
	let e;
	for (; je;) {
		let t = je;
		for (je = void 0; t;) {
			let n = t.next;
			if (t.next = void 0, t.flags &= -9, t.flags & 1) try {
				t.trigger();
			} catch (t) {
				e ||= t;
			}
			t = n;
		}
	}
	if (e) throw e;
}
function Ie(e) {
	for (let t = e.deps; t; t = t.nextDep) t.version = -1, t.prevActiveLink = t.dep.activeLink, t.dep.activeLink = t;
}
function Le(e) {
	let t, n = e.depsTail, r = n;
	for (; r;) {
		let e = r.prevDep;
		r.version === -1 ? (r === n && (n = e), Be(r), Ve(r)) : t = r, r.dep.activeLink = r.prevActiveLink, r.prevActiveLink = void 0, r = e;
	}
	e.deps = t, e.depsTail = n;
}
function Re(e) {
	for (let t = e.deps; t; t = t.nextDep) if (t.dep.version !== t.version || t.dep.computed && (ze(t.dep.computed) || t.dep.version !== t.version)) return !0;
	return !!e._dirty;
}
function ze(e) {
	if (e.flags & 4 && !(e.flags & 16) || (e.flags &= -17, e.globalVersion === qe) || (e.globalVersion = qe, !e.isSSR && e.flags & 128 && (!e.deps && !e._dirty || !Re(e)))) return;
	e.flags |= 2;
	let t = e.dep, n = j, r = He;
	j = e, He = !0;
	try {
		Ie(e);
		let n = e.fn(e._value);
		(t.version === 0 || oe(n, e._value)) && (e.flags |= 128, e._value = n, t.version++);
	} catch (e) {
		throw t.version++, e;
	} finally {
		j = n, He = r, Le(e), e.flags &= -3;
	}
}
function Be(e, t = !1) {
	let { dep: n, prevSub: r, nextSub: i } = e;
	if (r && (r.nextSub = i, e.prevSub = void 0), i && (i.prevSub = r, e.nextSub = void 0), n.subs === e && (n.subs = r, !r && n.computed)) {
		n.computed.flags &= -5;
		for (let e = n.computed.deps; e; e = e.nextDep) Be(e, !0);
	}
	!t && !--n.sc && n.map && n.map.delete(n.key);
}
function Ve(e) {
	let { prevDep: t, nextDep: n } = e;
	t && (t.nextDep = n, e.prevDep = void 0), n && (n.prevDep = t, e.nextDep = void 0);
}
var He = !0, Ue = [];
function We() {
	Ue.push(He), He = !1;
}
function Ge() {
	let e = Ue.pop();
	He = e === void 0 || e;
}
function Ke(e) {
	let { cleanup: t } = e;
	if (e.cleanup = void 0, t) {
		let e = j;
		j = void 0;
		try {
			t();
		} finally {
			j = e;
		}
	}
}
var qe = 0, Je = class {
	constructor(e, t) {
		this.sub = e, this.dep = t, this.version = t.version, this.nextDep = this.prevDep = this.nextSub = this.prevSub = this.prevActiveLink = void 0;
	}
}, Ye = class {
	constructor(e) {
		this.computed = e, this.version = 0, this.activeLink = void 0, this.subs = void 0, this.map = void 0, this.key = void 0, this.sc = 0, this.__v_skip = !0;
	}
	track(e) {
		if (!j || !He || j === this.computed) return;
		let t = this.activeLink;
		if (t === void 0 || t.sub !== j) t = this.activeLink = new Je(j, this), j.deps ? (t.prevDep = j.depsTail, j.depsTail.nextDep = t, j.depsTail = t) : j.deps = j.depsTail = t, Xe(t);
		else if (t.version === -1 && (t.version = this.version, t.nextDep)) {
			let e = t.nextDep;
			e.prevDep = t.prevDep, t.prevDep && (t.prevDep.nextDep = e), t.prevDep = j.depsTail, t.nextDep = void 0, j.depsTail.nextDep = t, j.depsTail = t, j.deps === t && (j.deps = e);
		}
		return t;
	}
	trigger(e) {
		this.version++, qe++, this.notify(e);
	}
	notify(e) {
		Pe();
		try {
			for (let e = this.subs; e; e = e.prevSub) e.sub.notify() && e.sub.dep.notify();
		} finally {
			Fe();
		}
	}
};
function Xe(e) {
	if (e.dep.sc++, e.sub.flags & 4) {
		let t = e.dep.computed;
		if (t && !e.dep.subs) {
			t.flags |= 20;
			for (let e = t.deps; e; e = e.nextDep) Xe(e);
		}
		let n = e.dep.subs;
		n !== e && (e.prevSub = n, n && (n.nextSub = e)), e.dep.subs = e;
	}
}
var Ze = /* @__PURE__ */ new WeakMap(), Qe = /* @__PURE__ */ Symbol(""), $e = /* @__PURE__ */ Symbol(""), et = /* @__PURE__ */ Symbol("");
function M(e, t, n) {
	if (He && j) {
		let t = Ze.get(e);
		t || Ze.set(e, t = /* @__PURE__ */ new Map());
		let r = t.get(n);
		r || (t.set(n, r = new Ye()), r.map = t, r.key = n), r.track();
	}
}
function tt(e, t, n, r, i, a) {
	let o = Ze.get(e);
	if (!o) {
		qe++;
		return;
	}
	let s = (e) => {
		e && e.trigger();
	};
	if (Pe(), t === "clear") o.forEach(s);
	else {
		let i = d(e), a = i && w(n);
		if (i && n === "length") {
			let e = Number(r);
			o.forEach((t, n) => {
				(n === "length" || n === et || !_(n) && n >= e) && s(t);
			});
		} else switch ((n !== void 0 || o.has(void 0)) && s(o.get(n)), a && s(o.get(et)), t) {
			case "add":
				i ? a && s(o.get("length")) : (s(o.get(Qe)), f(e) && s(o.get($e)));
				break;
			case "delete":
				i || (s(o.get(Qe)), f(e) && s(o.get($e)));
				break;
			case "set": f(e) && s(o.get(Qe));
		}
	}
	Fe();
}
function nt(e) {
	let t = /* @__PURE__ */ N(e);
	return t === e ? t : (M(t, "iterate", et), /* @__PURE__ */ Vt(e) ? t : t.map(Wt));
}
function rt(e) {
	return M(e = /* @__PURE__ */ N(e), "iterate", et), e;
}
function it(e, t) {
	return /* @__PURE__ */ Bt(e) ? Gt(/* @__PURE__ */ zt(e) ? Wt(t) : t) : Wt(t);
}
var at = {
	__proto__: null,
	[Symbol.iterator]() {
		return ot(this, Symbol.iterator, (e) => it(this, e));
	},
	concat(...e) {
		return nt(this).concat(...e.map((e) => d(e) ? nt(e) : e));
	},
	entries() {
		return ot(this, "entries", (e) => (e[1] = it(this, e[1]), e));
	},
	every(e, t) {
		return ct(this, "every", e, t, void 0, arguments);
	},
	filter(e, t) {
		return ct(this, "filter", e, t, (e) => e.map((e) => it(this, e)), arguments);
	},
	find(e, t) {
		return ct(this, "find", e, t, (e) => it(this, e), arguments);
	},
	findIndex(e, t) {
		return ct(this, "findIndex", e, t, void 0, arguments);
	},
	findLast(e, t) {
		return ct(this, "findLast", e, t, (e) => it(this, e), arguments);
	},
	findLastIndex(e, t) {
		return ct(this, "findLastIndex", e, t, void 0, arguments);
	},
	forEach(e, t) {
		return ct(this, "forEach", e, t, void 0, arguments);
	},
	includes(...e) {
		return ut(this, "includes", e);
	},
	indexOf(...e) {
		return ut(this, "indexOf", e);
	},
	join(e) {
		return nt(this).join(e);
	},
	lastIndexOf(...e) {
		return ut(this, "lastIndexOf", e);
	},
	map(e, t) {
		return ct(this, "map", e, t, void 0, arguments);
	},
	pop() {
		return dt(this, "pop");
	},
	push(...e) {
		return dt(this, "push", e);
	},
	reduce(e, ...t) {
		return lt(this, "reduce", e, t);
	},
	reduceRight(e, ...t) {
		return lt(this, "reduceRight", e, t);
	},
	shift() {
		return dt(this, "shift");
	},
	some(e, t) {
		return ct(this, "some", e, t, void 0, arguments);
	},
	splice(...e) {
		return dt(this, "splice", e);
	},
	toReversed() {
		return nt(this).toReversed();
	},
	toSorted(e) {
		return nt(this).toSorted(e);
	},
	toSpliced(...e) {
		return nt(this).toSpliced(...e);
	},
	unshift(...e) {
		return dt(this, "unshift", e);
	},
	values() {
		return ot(this, "values", (e) => it(this, e));
	}
};
function ot(e, t, n) {
	let r = rt(e), i = r[t]();
	return r !== e && !/* @__PURE__ */ Vt(e) && (i._next = i.next, i.next = () => {
		let e = i._next();
		return e.done || (e.value = n(e.value)), e;
	}), i;
}
var st = Array.prototype;
function ct(e, t, n, r, i, a) {
	let o = rt(e), s = o !== e && !/* @__PURE__ */ Vt(e), c = o[t];
	if (c !== st[t]) {
		let t = c.apply(e, a);
		return s ? Wt(t) : t;
	}
	let l = n;
	o !== e && (s ? l = function(t, r) {
		return n.call(this, it(e, t), r, e);
	} : n.length > 2 && (l = function(t, r) {
		return n.call(this, t, r, e);
	}));
	let u = c.call(o, l, r);
	return s && i ? i(u) : u;
}
function lt(e, t, n, r) {
	let i = rt(e), a = i !== e && !/* @__PURE__ */ Vt(e), o = n, s = !1;
	i !== e && (a ? (s = r.length === 0, o = function(t, r, i) {
		return s && (s = !1, t = it(e, t)), n.call(this, t, it(e, r), i, e);
	}) : n.length > 3 && (o = function(t, r, i) {
		return n.call(this, t, r, i, e);
	}));
	let c = i[t](o, ...r);
	return s ? it(e, c) : c;
}
function ut(e, t, n) {
	let r = /* @__PURE__ */ N(e);
	M(r, "iterate", et);
	let i = r[t](...n);
	return (i === -1 || i === !1) && /* @__PURE__ */ Ht(n[0]) ? (n[0] = /* @__PURE__ */ N(n[0]), r[t](...n)) : i;
}
function dt(e, t, n = []) {
	We(), Pe();
	let r = (/* @__PURE__ */ N(e))[t].apply(e, n);
	return Fe(), Ge(), r;
}
var ft = /* @__PURE__ */ e("__proto__,__v_isRef,__isVue"), pt = new Set(/* @__PURE__ */ Object.getOwnPropertyNames(Symbol).filter((e) => e !== "arguments" && e !== "caller").map((e) => Symbol[e]).filter(_));
function mt(e) {
	_(e) || (e = String(e));
	let t = /* @__PURE__ */ N(this);
	return M(t, "has", e), t.hasOwnProperty(e);
}
var ht = class {
	constructor(e = !1, t = !1) {
		this._isReadonly = e, this._isShallow = t;
	}
	get(e, t, n) {
		if (t === "__v_skip") return e.__v_skip;
		let r = this._isReadonly, i = this._isShallow;
		if (t === "__v_isReactive") return !r;
		if (t === "__v_isReadonly") return r;
		if (t === "__v_isShallow") return i;
		if (t === "__v_raw") return n === (r ? i ? Nt : Mt : i ? jt : At).get(e) || Object.getPrototypeOf(e) === Object.getPrototypeOf(n) ? e : void 0;
		let a = d(e);
		if (!r) {
			let e;
			if (a && (e = at[t])) return e;
			if (t === "hasOwnProperty") return mt;
		}
		let o = Reflect.get(e, t, /* @__PURE__ */ P(e) ? e : n);
		if ((_(t) ? pt.has(t) : ft(t)) || (r || M(e, "get", t), i)) return o;
		if (/* @__PURE__ */ P(o)) {
			let e = a && w(t) ? o : o.value;
			return r && v(e) ? /* @__PURE__ */ Lt(e) : e;
		}
		return v(o) ? r ? /* @__PURE__ */ Lt(o) : /* @__PURE__ */ Ft(o) : o;
	}
}, gt = class extends ht {
	constructor(e = !1) {
		super(!1, e);
	}
	set(e, t, n, r) {
		let i = e[t], a = d(e) && w(t);
		if (!this._isShallow) {
			let e = /* @__PURE__ */ Bt(i);
			if (!/* @__PURE__ */ Vt(n) && !/* @__PURE__ */ Bt(n) && (i = /* @__PURE__ */ N(i), n = /* @__PURE__ */ N(n)), !a && /* @__PURE__ */ P(i) && !/* @__PURE__ */ P(n)) return e || (i.value = n), !0;
		}
		let o = a ? Number(t) < e.length : u(e, t), s = Reflect.set(e, t, n, /* @__PURE__ */ P(e) ? e : r);
		return e === /* @__PURE__ */ N(r) && s && (o ? oe(n, i) && tt(e, "set", t, n, i) : tt(e, "add", t, n)), s;
	}
	deleteProperty(e, t) {
		let n = u(e, t), r = e[t], i = Reflect.deleteProperty(e, t);
		return i && n && tt(e, "delete", t, void 0, r), i;
	}
	has(e, t) {
		let n = Reflect.has(e, t);
		return (!_(t) || !pt.has(t)) && M(e, "has", t), n;
	}
	ownKeys(e) {
		return M(e, "iterate", d(e) ? "length" : Qe), Reflect.ownKeys(e);
	}
}, _t = class extends ht {
	constructor(e = !1) {
		super(!0, e);
	}
	set(e, t) {
		return !0;
	}
	deleteProperty(e, t) {
		return !0;
	}
}, vt = /* @__PURE__ */ new gt(), yt = /* @__PURE__ */ new _t(), bt = /* @__PURE__ */ new gt(!0), xt = (e) => e, St = (e) => Reflect.getPrototypeOf(e);
function Ct(e, t, n) {
	return function(...r) {
		let i = this.__v_raw, a = /* @__PURE__ */ N(i), o = f(a), c = e === "entries" || e === Symbol.iterator && o, l = e === "keys" && o, u = i[e](...r), d = n ? xt : t ? Gt : Wt;
		return !t && M(a, "iterate", l ? $e : Qe), s(Object.create(u), { next() {
			let { value: e, done: t } = u.next();
			return t ? {
				value: e,
				done: t
			} : {
				value: c ? [d(e[0]), d(e[1])] : d(e),
				done: t
			};
		} });
	};
}
function wt(e) {
	return function(...t) {
		return e === "delete" ? !1 : e === "clear" ? void 0 : this;
	};
}
function Tt(e, t) {
	let n = {
		get(n) {
			let r = this.__v_raw, i = /* @__PURE__ */ N(r), a = /* @__PURE__ */ N(n);
			e || (oe(n, a) && M(i, "get", n), M(i, "get", a));
			let { has: o } = St(i), s = t ? xt : e ? Gt : Wt;
			if (o.call(i, n)) return s(r.get(n));
			if (o.call(i, a)) return s(r.get(a));
			r !== i && r.get(n);
		},
		get size() {
			let t = this.__v_raw;
			return !e && M(/* @__PURE__ */ N(t), "iterate", Qe), t.size;
		},
		has(t) {
			let n = this.__v_raw, r = /* @__PURE__ */ N(n), i = /* @__PURE__ */ N(t);
			return e || (oe(t, i) && M(r, "has", t), M(r, "has", i)), t === i ? n.has(t) : n.has(t) || n.has(i);
		},
		forEach(n, r) {
			let i = this, a = i.__v_raw, o = /* @__PURE__ */ N(a), s = t ? xt : e ? Gt : Wt;
			return !e && M(o, "iterate", Qe), a.forEach((e, t) => n.call(r, s(e), s(t), i));
		}
	};
	return s(n, e ? {
		add: wt("add"),
		set: wt("set"),
		delete: wt("delete"),
		clear: wt("clear")
	} : {
		add(e) {
			let n = /* @__PURE__ */ N(this), r = St(n), i = /* @__PURE__ */ N(e), a = !t && !/* @__PURE__ */ Vt(e) && !/* @__PURE__ */ Bt(e) ? i : e;
			return r.has.call(n, a) || oe(e, a) && r.has.call(n, e) || oe(i, a) && r.has.call(n, i) || (n.add(a), tt(n, "add", a, a)), this;
		},
		set(e, n) {
			!t && !/* @__PURE__ */ Vt(n) && !/* @__PURE__ */ Bt(n) && (n = /* @__PURE__ */ N(n));
			let r = /* @__PURE__ */ N(this), { has: i, get: a } = St(r), o = i.call(r, e);
			o ||= (e = /* @__PURE__ */ N(e), i.call(r, e));
			let s = a.call(r, e);
			return r.set(e, n), o ? oe(n, s) && tt(r, "set", e, n, s) : tt(r, "add", e, n), this;
		},
		delete(e) {
			let t = /* @__PURE__ */ N(this), { has: n, get: r } = St(t), i = n.call(t, e);
			i ||= (e = /* @__PURE__ */ N(e), n.call(t, e));
			let a = r ? r.call(t, e) : void 0, o = t.delete(e);
			return i && tt(t, "delete", e, void 0, a), o;
		},
		clear() {
			let e = /* @__PURE__ */ N(this), t = e.size !== 0, n = e.clear();
			return t && tt(e, "clear", void 0, void 0, void 0), n;
		}
	}), [
		"keys",
		"values",
		"entries",
		Symbol.iterator
	].forEach((r) => {
		n[r] = Ct(r, e, t);
	}), n;
}
function Et(e, t) {
	let n = Tt(e, t);
	return (t, r, i) => r === "__v_isReactive" ? !e : r === "__v_isReadonly" ? e : r === "__v_raw" ? t : Reflect.get(u(n, r) && r in t ? n : t, r, i);
}
var Dt = { get: /* @__PURE__ */ Et(!1, !1) }, Ot = { get: /* @__PURE__ */ Et(!1, !0) }, kt = { get: /* @__PURE__ */ Et(!0, !1) }, At = /* @__PURE__ */ new WeakMap(), jt = /* @__PURE__ */ new WeakMap(), Mt = /* @__PURE__ */ new WeakMap(), Nt = /* @__PURE__ */ new WeakMap();
function Pt(e) {
	switch (e) {
		case "Object":
		case "Array": return 1;
		case "Map":
		case "Set":
		case "WeakMap":
		case "WeakSet": return 2;
		default: return 0;
	}
}
// @__NO_SIDE_EFFECTS__
function Ft(e) {
	return /* @__PURE__ */ Bt(e) ? e : Rt(e, !1, vt, Dt, At);
}
// @__NO_SIDE_EFFECTS__
function It(e) {
	return Rt(e, !1, bt, Ot, jt);
}
// @__NO_SIDE_EFFECTS__
function Lt(e) {
	return Rt(e, !0, yt, kt, Mt);
}
function Rt(e, t, n, r, i) {
	if (!v(e) || e.__v_raw && !(t && e.__v_isReactive) || e.__v_skip || !Object.isExtensible(e)) return e;
	let a = i.get(e);
	if (a) return a;
	let o = Pt(S(e));
	if (o === 0) return e;
	let s = new Proxy(e, o === 2 ? r : n);
	return i.set(e, s), s;
}
// @__NO_SIDE_EFFECTS__
function zt(e) {
	return /* @__PURE__ */ Bt(e) ? /* @__PURE__ */ zt(e.__v_raw) : !!(e && e.__v_isReactive);
}
// @__NO_SIDE_EFFECTS__
function Bt(e) {
	return !!(e && e.__v_isReadonly);
}
// @__NO_SIDE_EFFECTS__
function Vt(e) {
	return !!(e && e.__v_isShallow);
}
// @__NO_SIDE_EFFECTS__
function Ht(e) {
	return e ? !!e.__v_raw : !1;
}
// @__NO_SIDE_EFFECTS__
function N(e) {
	let t = e && e.__v_raw;
	return t ? /* @__PURE__ */ N(t) : e;
}
function Ut(e) {
	return !u(e, "__v_skip") && Object.isExtensible(e) && D(e, "__v_skip", !0), e;
}
var Wt = (e) => v(e) ? /* @__PURE__ */ Ft(e) : e, Gt = (e) => v(e) ? /* @__PURE__ */ Lt(e) : e;
// @__NO_SIDE_EFFECTS__
function P(e) {
	return e ? e.__v_isRef === !0 : !1;
}
// @__NO_SIDE_EFFECTS__
function F(e) {
	return Kt(e, !1);
}
function Kt(e, t) {
	return /* @__PURE__ */ P(e) ? e : new qt(e, t);
}
var qt = class {
	constructor(e, t) {
		this.dep = new Ye(), this.__v_isRef = !0, this.__v_isShallow = !1, this._rawValue = t ? e : /* @__PURE__ */ N(e), this._value = t ? e : Wt(e), this.__v_isShallow = t;
	}
	get value() {
		return this.dep.track(), this._value;
	}
	set value(e) {
		let t = this._rawValue, n = this.__v_isShallow || /* @__PURE__ */ Vt(e) || /* @__PURE__ */ Bt(e);
		e = n ? e : /* @__PURE__ */ N(e), oe(e, t) && (this._rawValue = e, this._value = n ? e : Wt(e), this.dep.trigger());
	}
};
function I(e) {
	return /* @__PURE__ */ P(e) ? e.value : e;
}
var Jt = {
	get: (e, t, n) => t === "__v_raw" ? e : I(Reflect.get(e, t, n)),
	set: (e, t, n, r) => {
		let i = e[t];
		return /* @__PURE__ */ P(i) && !/* @__PURE__ */ P(n) ? (i.value = n, !0) : Reflect.set(e, t, n, r);
	}
};
function Yt(e) {
	return /* @__PURE__ */ zt(e) ? e : new Proxy(e, Jt);
}
var Xt = class {
	constructor(e, t, n) {
		this.fn = e, this.setter = t, this._value = void 0, this.dep = new Ye(this), this.__v_isRef = !0, this.deps = void 0, this.depsTail = void 0, this.flags = 16, this.globalVersion = qe - 1, this.next = void 0, this.effect = this, this.__v_isReadonly = !t, this.isSSR = n;
	}
	notify() {
		if (this.flags |= 16, !(this.flags & 8) && j !== this) return Ne(this, !0), !0;
	}
	get value() {
		let e = this.dep.track();
		return ze(this), e && (e.version = this.dep.version), this._value;
	}
	set value(e) {
		this.setter && this.setter(e);
	}
};
// @__NO_SIDE_EFFECTS__
function Zt(e, t, n = !1) {
	let r, i;
	return h(e) ? r = e : (r = e.get, i = e.set), new Xt(r, i, n);
}
var Qt = {}, $t = /* @__PURE__ */ new WeakMap(), en = void 0;
function tn(e, t = !1, n = en) {
	if (n) {
		let t = $t.get(n);
		t || $t.set(n, t = []), t.push(e);
	}
}
function nn(e, n, i = t) {
	let { immediate: a, deep: o, once: s, scheduler: l, augmentJob: u, call: f } = i, p = (e) => o ? e : /* @__PURE__ */ Vt(e) || o === !1 || o === 0 ? rn(e, 1) : rn(e), m, g, _, v, y = !1, b = !1;
	if (/* @__PURE__ */ P(e) ? (g = () => e.value, y = /* @__PURE__ */ Vt(e)) : /* @__PURE__ */ zt(e) ? (g = () => p(e), y = !0) : d(e) ? (b = !0, y = e.some((e) => /* @__PURE__ */ zt(e) || /* @__PURE__ */ Vt(e)), g = () => e.map((e) => {
		if (/* @__PURE__ */ P(e)) return e.value;
		if (/* @__PURE__ */ zt(e)) return p(e);
		if (h(e)) return f ? f(e, 2) : e();
	})) : g = h(e) ? n ? f ? () => f(e, 2) : e : () => {
		if (_) {
			We();
			try {
				_();
			} finally {
				Ge();
			}
		}
		let t = en;
		en = m;
		try {
			return f ? f(e, 3, [v]) : e(v);
		} finally {
			en = t;
		}
	} : r, n && o) {
		let e = g, t = o === !0 ? Infinity : o;
		g = () => rn(e(), t);
	}
	let x = De(), S = () => {
		m.stop(), x && x.active && c(x.effects, m);
	};
	if (s && n) {
		let e = n;
		n = (...t) => {
			let n = e(...t);
			return S(), n;
		};
	}
	let C = b ? Array(e.length).fill(Qt) : Qt, w = (e) => {
		if (!(!(m.flags & 1) || !m.dirty && !e)) if (n) {
			let t = m.run();
			if (e || o || y || (b ? t.some((e, t) => oe(e, C[t])) : oe(t, C))) {
				_ && _();
				let e = en;
				en = m;
				try {
					let e = [
						t,
						C === Qt ? void 0 : b && C[0] === Qt ? [] : C,
						v
					];
					C = t, f ? f(n, 3, e) : n(...e);
				} finally {
					en = e;
				}
			}
		} else m.run();
	};
	return u && u(w), m = new ke(g), m.scheduler = l ? () => l(w, !1) : w, v = (e) => tn(e, !1, m), _ = m.onStop = () => {
		let e = $t.get(m);
		if (e) {
			if (f) f(e, 4);
			else for (let t of e) t();
			$t.delete(m);
		}
	}, n ? a ? w(!0) : C = m.run() : l ? l(w.bind(null, !0), !0) : m.run(), S.pause = m.pause.bind(m), S.resume = m.resume.bind(m), S.stop = S, S;
}
function rn(e, t = Infinity, n) {
	if (t <= 0 || !v(e) || e.__v_skip || (n ||= /* @__PURE__ */ new Map(), (n.get(e) || 0) >= t)) return e;
	if (n.set(e, t), t--, /* @__PURE__ */ P(e)) rn(e.value, t, n);
	else if (d(e)) for (let r = 0; r < e.length; r++) rn(e[r], t, n);
	else if (p(e) || f(e)) e.forEach((e) => {
		rn(e, t, n);
	});
	else if (C(e)) {
		for (let r in e) rn(e[r], t, n);
		for (let r of Object.getOwnPropertySymbols(e)) Object.prototype.propertyIsEnumerable.call(e, r) && rn(e[r], t, n);
	}
	return e;
}
//#endregion
//#region node_modules/@vue/runtime-core/dist/runtime-core.esm-bundler.js
function an(e, t, n, r) {
	try {
		return r ? e(...r) : e();
	} catch (e) {
		sn(e, t, n);
	}
}
function on(e, t, n, r) {
	if (h(e)) {
		let i = an(e, t, n, r);
		return i && y(i) && i.catch((e) => {
			sn(e, t, n);
		}), i;
	}
	if (d(e)) {
		let i = [];
		for (let a = 0; a < e.length; a++) i.push(on(e[a], t, n, r));
		return i;
	}
}
function sn(e, n, r, i = !0) {
	let a = n ? n.vnode : null, { errorHandler: o, throwUnhandledErrorInProduction: s } = n && n.appContext.config || t;
	if (n) {
		let t = n.parent, i = n.proxy, a = `https://vuejs.org/error-reference/#runtime-${r}`;
		for (; t;) {
			let n = t.ec;
			if (n) {
				for (let t = 0; t < n.length; t++) if (n[t](e, i, a) === !1) return;
			}
			t = t.parent;
		}
		if (o) {
			We(), an(o, null, 10, [
				e,
				i,
				a
			]), Ge();
			return;
		}
	}
	cn(e, r, a, i, s);
}
function cn(e, t, n, r = !0, i = !1) {
	if (i) throw e;
	console.error(e);
}
var L = [], ln = -1, un = [], dn = null, fn = 0, pn = /* @__PURE__ */ Promise.resolve(), mn = null;
function hn(e) {
	let t = mn || pn;
	return e ? t.then(this ? e.bind(this) : e) : t;
}
function gn(e) {
	let t = ln + 1, n = L.length;
	for (; t < n;) {
		let r = t + n >>> 1, i = L[r], a = Sn(i);
		a < e || a === e && i.flags & 2 ? t = r + 1 : n = r;
	}
	return t;
}
function _n(e) {
	if (!(e.flags & 1)) {
		let t = Sn(e), n = L[L.length - 1];
		!n || !(e.flags & 2) && t >= Sn(n) ? L.push(e) : L.splice(gn(t), 0, e), e.flags |= 1, vn();
	}
}
function vn() {
	mn ||= pn.then(Cn);
}
function yn(e) {
	d(e) ? un.push(...e) : dn && e.id === -1 ? dn.splice(fn + 1, 0, e) : e.flags & 1 || (un.push(e), e.flags |= 1), vn();
}
function bn(e, t, n = ln + 1) {
	for (; n < L.length; n++) {
		let t = L[n];
		if (t && t.flags & 2) {
			if (e && t.id !== e.uid) continue;
			L.splice(n, 1), n--, t.flags & 4 && (t.flags &= -2), t(), t.flags & 4 || (t.flags &= -2);
		}
	}
}
function xn(e) {
	if (un.length) {
		let e = [...new Set(un)].sort((e, t) => Sn(e) - Sn(t));
		if (un.length = 0, dn) {
			dn.push(...e);
			return;
		}
		for (dn = e, fn = 0; fn < dn.length; fn++) {
			let e = dn[fn];
			e.flags & 4 && (e.flags &= -2), e.flags & 8 || e(), e.flags &= -2;
		}
		dn = null, fn = 0;
	}
}
var Sn = (e) => e.id == null ? e.flags & 2 ? -1 : Infinity : e.id;
function Cn(e) {
	try {
		for (ln = 0; ln < L.length; ln++) {
			let e = L[ln];
			e && !(e.flags & 8) && (e.flags & 4 && (e.flags &= -2), an(e, e.i, e.i ? 15 : 14), e.flags & 4 || (e.flags &= -2));
		}
	} finally {
		for (; ln < L.length; ln++) {
			let e = L[ln];
			e && (e.flags &= -2);
		}
		ln = -1, L.length = 0, xn(e), mn = null, (L.length || un.length) && Cn(e);
	}
}
var wn = null, Tn = null;
function En(e) {
	let t = wn;
	return wn = e, Tn = e && e.type.__scopeId || null, t;
}
function Dn(e, t = wn, n) {
	if (!t || e._n) return e;
	let r = (...n) => {
		r._d && ta(-1);
		let i = En(t), a = Zi.length, o;
		try {
			o = e(...n);
		} finally {
			for (let e = Zi.length; e > a; e--) $i();
			En(i), r._d && ta(1);
		}
		return o;
	};
	return r._n = !0, r._c = !0, r._d = !0, r;
}
function R(e, n) {
	if (wn === null) return e;
	let r = Ia(wn), i = e.dirs ||= [];
	for (let e = 0; e < n.length; e++) {
		let [a, o, s, c = t] = n[e];
		a && (h(a) && (a = {
			mounted: a,
			updated: a
		}), a.deep && rn(o), i.push({
			dir: a,
			instance: r,
			value: o,
			oldValue: void 0,
			arg: s,
			modifiers: c
		}));
	}
	return e;
}
function On(e, t, n, r) {
	let i = e.dirs, a = t && t.dirs;
	for (let o = 0; o < i.length; o++) {
		let s = i[o];
		a && (s.oldValue = a[o].value);
		let c = s.dir[r];
		c && (We(), on(c, n, 8, [
			e.el,
			s,
			e,
			t
		]), Ge());
	}
}
function kn(e, t) {
	if (J) {
		let n = J.provides, r = J.parent && J.parent.provides;
		r === n && (n = J.provides = Object.create(r)), n[e] = t;
	}
}
function An(e, t, n = !1) {
	let r = xa();
	if (r || oi) {
		let i = oi ? oi._context.provides : r ? r.parent == null || r.ce ? r.vnode.appContext && r.vnode.appContext.provides : r.parent.provides : void 0;
		if (i && e in i) return i[e];
		if (arguments.length > 1) return n && h(t) ? t.call(r && r.proxy) : t;
	}
}
var jn = /* @__PURE__ */ Symbol.for("v-scx"), Mn = () => An(jn);
function Nn(e, t, n) {
	return Pn(e, t, n);
}
function Pn(e, n, i = t) {
	let { immediate: a, deep: o, flush: c, once: l } = i, u = s({}, i), d = n && a || !n && c !== "post", f;
	if (Da) {
		if (c === "sync") {
			let e = Mn();
			f = e.__watcherHandles ||= [];
		} else if (!d) {
			let e = () => {};
			return e.stop = r, e.resume = r, e.pause = r, e;
		}
	}
	let p = J;
	u.call = (e, t, n) => on(e, p, t, n);
	let m = !1;
	c === "post" ? u.scheduler = (e) => {
		B(e, p && p.suspense);
	} : c !== "sync" && (m = !0, u.scheduler = (e, t) => {
		t ? e() : _n(e);
	}), u.augmentJob = (e) => {
		n && (e.flags |= 4), m && (e.flags |= 2, p && (e.id = p.uid, e.i = p));
	};
	let h = nn(e, n, u);
	return Da && (f ? f.push(h) : d && h()), h;
}
function Fn(e, t, n) {
	let r = this.proxy, i = g(e) ? e.includes(".") ? In(r, e) : () => r[e] : e.bind(r, r), a;
	h(t) ? a = t : (a = t.handler, n = t);
	let o = wa(this), s = Pn(i, a.bind(r), n);
	return o(), s;
}
function In(e, t) {
	let n = t.split(".");
	return () => {
		let t = e;
		for (let e = 0; e < n.length && t; e++) t = t[n[e]];
		return t;
	};
}
var Ln = /* @__PURE__ */ new WeakMap(), Rn = /* @__PURE__ */ Symbol("_vte"), zn = (e) => e.__isTeleport, Bn = (e) => e && (e.disabled || e.disabled === ""), Vn = (e) => e && (e.defer || e.defer === ""), Hn = (e) => typeof SVGElement < "u" && e instanceof SVGElement, Un = (e) => typeof MathMLElement == "function" && e instanceof MathMLElement, Wn = (e, t) => {
	let n = e && e.to;
	return g(n) ? t ? t(n) : null : n;
}, Gn = {
	name: "Teleport",
	__isTeleport: !0,
	process(e, t, n, r, i, a, o, s, c, l) {
		let { mc: u, pc: d, pbc: f, o: { insert: p, querySelector: m, createText: h, createComment: g, parentNode: _ } } = l, v = Bn(t.props), { dynamicChildren: y } = t, b = (e, t, n) => {
			e.shapeFlag & 16 && u(e.children, t, n, i, a, o, s, c);
		}, x = (e = t) => {
			let n = Bn(e.props), r = e.target = Wn(e.props, m), a = Xn(r, e, h, p);
			r && (o !== "svg" && Hn(r) ? o = "svg" : o !== "mathml" && Un(r) && (o = "mathml"), i && i.isCE && (i.ce._teleportTargets || (i.ce._teleportTargets = /* @__PURE__ */ new Set())).add(r), n || (b(e, r, a), Yn(e, !1)));
		}, S = (e) => {
			let t = () => {
				if (Ln.get(e) === t) {
					if (Ln.delete(e), Bn(e.props)) {
						let t = _(e.el) || n;
						b(e, t, e.anchor), Yn(e, !0);
					}
					x(e);
				}
			};
			Ln.set(e, t), B(t, a);
		};
		if (e == null) {
			let e = t.el = h(""), i = t.anchor = h("");
			if (p(e, n, r), p(i, n, r), Vn(t.props) || a && a.pendingBranch) {
				S(t);
				return;
			}
			v && (b(t, n, i), Yn(t, !0)), x();
		} else {
			t.el = e.el;
			let r = t.anchor = e.anchor, u = Ln.get(e);
			if (u) {
				u.flags |= 8, Ln.delete(e), S(t);
				return;
			}
			t.targetStart = e.targetStart;
			let p = t.target = e.target, h = t.targetAnchor = e.targetAnchor, g = Bn(e.props), _ = g ? n : p, b = g ? r : h;
			if (o === "svg" || Hn(p) ? o = "svg" : (o === "mathml" || Un(p)) && (o = "mathml"), y ? (f(e.dynamicChildren, y, _, i, a, o, s), Hi(e, t, !0)) : c || d(e, t, _, b, i, a, o, s, !1), v) g ? t.props && e.props && t.props.to !== e.props.to && (t.props.to = e.props.to) : Kn(t, n, r, l, 1);
			else if ((t.props && t.props.to) !== (e.props && e.props.to)) {
				let e = Wn(t.props, m);
				e && (t.target = e, Kn(t, e, null, l, 0));
			} else g && Kn(t, p, h, l, 1);
			Yn(t, v);
		}
	},
	remove(e, t, n, { um: r, o: { remove: i } }, a) {
		let { shapeFlag: o, children: s, anchor: c, targetStart: l, targetAnchor: u, target: d, props: f } = e, p = Bn(f), m = a || !p, h = Ln.get(e);
		if (h && (h.flags |= 8, Ln.delete(e)), d && (i(l), i(u)), a && i(c), !h && (p || d) && o & 16) for (let e = 0; e < s.length; e++) {
			let i = s[e];
			r(i, t, n, m, !!i.dynamicChildren);
		}
	},
	move: Kn,
	hydrate: qn
};
function Kn(e, t, n, { o: { insert: r }, m: i }, a = 2) {
	a === 0 && r(e.targetAnchor, t, n);
	let { el: o, anchor: s, shapeFlag: c, children: l, props: u } = e, d = a === 2;
	if (d && r(o, t, n), !Ln.has(e) && (!d || Bn(u)) && c & 16) for (let e = 0; e < l.length; e++) i(l[e], t, n, 2);
	d && r(s, t, n);
}
function qn(e, t, n, r, i, a, { o: { nextSibling: o, parentNode: s, querySelector: c, insert: l, createText: u } }, d) {
	function f(e, n) {
		let r = n;
		for (; r;) {
			if (r && r.nodeType === 8) {
				if (r.data === "teleport start anchor") t.targetStart = r;
				else if (r.data === "teleport anchor") {
					t.targetAnchor = r, e._lpa = t.targetAnchor && o(t.targetAnchor);
					break;
				}
			}
			r = o(r);
		}
	}
	function p(e, t) {
		t.anchor = d(o(e), t, s(e), n, r, i, a);
	}
	let m = t.target = Wn(t.props, c), h = Bn(t.props);
	if (m) {
		let c = m._lpa || m.firstChild;
		t.shapeFlag & 16 && (h ? (p(e, t), f(m, c), t.targetAnchor || Xn(m, t, u, l, s(e) === m ? e : null)) : (t.anchor = o(e), f(m, c), t.targetAnchor || Xn(m, t, u, l), d(c && o(c), t, m, n, r, i, a))), Yn(t, h);
	} else h && t.shapeFlag & 16 && (p(e, t), t.targetStart = e, t.targetAnchor = o(e));
	return t.anchor && o(t.anchor);
}
var Jn = Gn;
function Yn(e, t) {
	let n = e.ctx;
	if (n && n.ut) {
		let r, i;
		for (t ? (r = e.el, i = e.anchor) : (r = e.targetStart, i = e.targetAnchor); r && r !== i;) r.nodeType === 1 && r.setAttribute("data-v-owner", n.uid), r = r.nextSibling;
		n.ut();
	}
}
function Xn(e, t, n, r, i = null) {
	let a = t.targetStart = n(""), o = t.targetAnchor = n("");
	return a[Rn] = o, e && (r(a, e, i), r(o, e, i)), o;
}
var Zn = /* @__PURE__ */ Symbol("_leaveCb"), Qn = /* @__PURE__ */ Symbol("_enterCb");
function $n() {
	let e = {
		isMounted: !1,
		isLeaving: !1,
		isUnmounting: !1,
		leavingVNodes: /* @__PURE__ */ new Map()
	};
	return Dr(() => {
		e.isMounted = !0;
	}), Ar(() => {
		e.isUnmounting = !0;
	}), e;
}
var er = [Function, Array], tr = {
	mode: String,
	appear: Boolean,
	persisted: Boolean,
	onBeforeEnter: er,
	onEnter: er,
	onAfterEnter: er,
	onEnterCancelled: er,
	onBeforeLeave: er,
	onLeave: er,
	onAfterLeave: er,
	onLeaveCancelled: er,
	onBeforeAppear: er,
	onAppear: er,
	onAfterAppear: er,
	onAppearCancelled: er
}, nr = (e) => {
	let t = e.subTree;
	return t.component ? nr(t.component) : t;
}, rr = {
	name: "BaseTransition",
	props: tr,
	setup(e, { slots: t }) {
		let n = xa(), r = $n();
		return () => {
			let i = t.default && dr(t.default(), !0), a = i && i.length ? ir(i) : n.subTree ? q() : void 0;
			if (!a) return;
			let o = /* @__PURE__ */ N(e), { mode: s } = o;
			if (r.isLeaving) return cr(a);
			let c = lr(a);
			if (!c) return cr(a);
			let l = sr(c, o, r, n, (e) => l = e);
			c.type !== H && ur(c, l);
			let u = n.subTree && lr(n.subTree);
			if (u && u.type !== H && !aa(u, c) && nr(n).type !== H) {
				let e = sr(u, o, r, n);
				if (ur(u, e), s === "out-in" && c.type !== H) return r.isLeaving = !0, e.afterLeave = () => {
					r.isLeaving = !1, n.job.flags & 8 || n.update(), delete e.afterLeave, u = void 0;
				}, cr(a);
				s === "in-out" && c.type !== H ? e.delayLeave = (e, t, n) => {
					let i = or(r, u);
					i[String(u.key)] = u, e[Zn] = () => {
						t(), e[Zn] = void 0, delete l.delayedLeave, u = void 0;
					}, l.delayedLeave = () => {
						n(), delete l.delayedLeave, u = void 0;
					};
				} : u = void 0;
			} else u &&= void 0;
			return a;
		};
	}
};
function ir(e) {
	let t = e[0];
	if (e.length > 1) {
		for (let n of e) if (n.type !== H) {
			t = n;
			break;
		}
	}
	return t;
}
var ar = rr;
function or(e, t) {
	let { leavingVNodes: n } = e, r = n.get(t.type);
	return r || (r = /* @__PURE__ */ Object.create(null), n.set(t.type, r)), r;
}
function sr(e, t, n, r, i) {
	let { appear: a, mode: o, persisted: s = !1, onBeforeEnter: c, onEnter: l, onAfterEnter: u, onEnterCancelled: f, onBeforeLeave: p, onLeave: m, onAfterLeave: h, onLeaveCancelled: g, onBeforeAppear: _, onAppear: v, onAfterAppear: y, onAppearCancelled: b } = t, x = String(e.key), S = or(n, e), C = (e, t) => {
		e && on(e, r, 9, t);
	}, w = (e, t) => {
		let n = t[1];
		C(e, t), d(e) ? e.every((e) => e.length <= 1) && n() : e.length <= 1 && n();
	}, ee = {
		mode: o,
		persisted: s,
		beforeEnter(t) {
			let r = c;
			if (!n.isMounted) if (a) r = _ || c;
			else return;
			t[Zn] && t[Zn](!0);
			let i = S[x];
			i && aa(e, i) && i.el[Zn] && i.el[Zn](), C(r, [t]);
		},
		enter(t) {
			if (S[x] === e) return;
			let r = l, i = u, o = f;
			if (!n.isMounted) if (a) r = v || l, i = y || u, o = b || f;
			else return;
			let s = !1;
			t[Qn] = (e) => {
				s || (s = !0, C(e ? o : i, [t]), ee.delayedLeave && ee.delayedLeave(), t[Qn] = void 0);
			};
			let c = t[Qn].bind(null, !1);
			r ? w(r, [t, c]) : c();
		},
		leave(t, r) {
			let i = String(e.key);
			if (t[Qn] && t[Qn](!0), n.isUnmounting) return r();
			C(p, [t]);
			let a = !1;
			t[Zn] = (n) => {
				a || (a = !0, r(), C(n ? g : h, [t]), t[Zn] = void 0, S[i] === e && delete S[i]);
			};
			let o = t[Zn].bind(null, !1);
			S[i] = e, m ? w(m, [t, o]) : o();
		},
		clone(e) {
			let a = sr(e, t, n, r, i);
			return i && i(a), a;
		}
	};
	return ee;
}
function cr(e) {
	if (yr(e)) return e = ua(e), e.children = null, e;
}
function lr(e) {
	if (!yr(e)) return zn(e.type) && e.children ? ir(e.children) : e;
	if (e.component) return e.component.subTree;
	let { shapeFlag: t, children: n } = e;
	if (n) {
		if (t & 16) return n[0];
		if (t & 32 && h(n.default)) return n.default();
	}
}
function ur(e, t) {
	e.shapeFlag & 6 && e.component ? (e.transition = t, ur(e.component.subTree, t)) : e.shapeFlag & 128 ? (e.ssContent.transition = t.clone(e.ssContent), e.ssFallback.transition = t.clone(e.ssFallback)) : e.transition = t;
}
function dr(e, t = !1, n) {
	let r = [], i = 0;
	for (let a = 0; a < e.length; a++) {
		let o = e[a], s = n == null ? o.key : String(n) + String(o.key == null ? a : o.key);
		o.type === V ? (o.patchFlag & 128 && i++, r = r.concat(dr(o.children, t, s))) : (t || o.type !== H) && r.push(s == null ? o : ua(o, { key: s }));
	}
	if (i > 1) for (let e = 0; e < r.length; e++) r[e].patchFlag = -2;
	return r;
}
// @__NO_SIDE_EFFECTS__
function fr(e, t) {
	return h(e) ? /* @__PURE__ */ s({ name: e.name }, t, { setup: e }) : e;
}
function pr(e) {
	e.ids = [
		e.ids[0] + e.ids[2]++ + "-",
		0,
		0
	];
}
function mr(e, t) {
	let n;
	return !!((n = Object.getOwnPropertyDescriptor(e, t)) && !n.configurable);
}
var hr = /* @__PURE__ */ new WeakMap();
function gr(e, n, r, a, o = !1) {
	if (d(e)) {
		e.forEach((e, t) => gr(e, n && (d(n) ? n[t] : n), r, a, o));
		return;
	}
	if (vr(a) && !o) {
		a.shapeFlag & 512 && a.type.__asyncResolved && a.component.subTree.component && gr(e, n, r, a.component.subTree);
		return;
	}
	let s = a.shapeFlag & 4 ? Ia(a.component) : a.el, l = o ? null : s, { i: f, r: p } = e, m = n && n.r, _ = f.refs === t ? f.refs = {} : f.refs, v = f.setupState, y = /* @__PURE__ */ N(v), b = v === t ? i : (e) => !mr(_, e) && u(y, e), x = (e, t) => !(t && mr(_, t));
	if (m != null && m !== p) {
		if (_r(n), g(m)) _[m] = null, b(m) && (v[m] = null);
		else if (/* @__PURE__ */ P(m)) {
			let e = n;
			x(m, e.k) && (m.value = null), e.k && (_[e.k] = null);
		}
	}
	if (h(p)) an(p, f, 12, [l, _]);
	else {
		let t = g(p), n = /* @__PURE__ */ P(p);
		if (t || n) {
			let i = () => {
				if (e.f) {
					let n = t ? b(p) ? v[p] : _[p] : x(p) || !e.k ? p.value : _[e.k];
					if (o) d(n) && c(n, s);
					else if (d(n)) n.includes(s) || n.push(s);
					else if (t) _[p] = [s], b(p) && (v[p] = _[p]);
					else {
						let t = [s];
						x(p, e.k) && (p.value = t), e.k && (_[e.k] = t);
					}
				} else t ? (_[p] = l, b(p) && (v[p] = l)) : n && (x(p, e.k) && (p.value = l), e.k && (_[e.k] = l));
			};
			if (l) {
				let t = () => {
					i(), hr.delete(e);
				};
				t.id = -1, hr.set(e, t), B(t, r);
			} else _r(e), i();
		}
	}
}
function _r(e) {
	let t = hr.get(e);
	t && (t.flags |= 8, hr.delete(e));
}
de().requestIdleCallback, de().cancelIdleCallback;
var vr = (e) => !!e.type.__asyncLoader, yr = (e) => e.type.__isKeepAlive;
function br(e, t) {
	Sr(e, "a", t);
}
function xr(e, t) {
	Sr(e, "da", t);
}
function Sr(e, t, n = J) {
	let r = e.__wdc ||= () => {
		let t = n;
		for (; t;) {
			if (t.isDeactivated) return;
			t = t.parent;
		}
		return e();
	};
	if (wr(t, r, n), n) {
		let e = n.parent;
		for (; e && e.parent;) yr(e.parent.vnode) && Cr(r, t, n, e), e = e.parent;
	}
}
function Cr(e, t, n, r) {
	let i = wr(t, e, r, !0);
	jr(() => {
		c(r[t], i);
	}, n);
}
function wr(e, t, n = J, r = !1) {
	if (n) {
		let i = n[e] || (n[e] = []), a = t.__weh ||= (...r) => {
			We();
			let i = wa(n), a = on(t, n, e, r);
			return i(), Ge(), a;
		};
		return r ? i.unshift(a) : i.push(a), a;
	}
}
var Tr = (e) => (t, n = J) => {
	(!Da || e === "sp") && wr(e, (...e) => t(...e), n);
}, Er = Tr("bm"), Dr = Tr("m"), Or = Tr("bu"), kr = Tr("u"), Ar = Tr("bum"), jr = Tr("um"), Mr = Tr("sp"), Nr = Tr("rtg"), Pr = Tr("rtc");
function Fr(e, t = J) {
	wr("ec", e, t);
}
var Ir = /* @__PURE__ */ Symbol.for("v-ndc");
function Lr(e, t, n, r) {
	let i, a = n && n[r], o = d(e);
	if (o || g(e)) {
		let n = o && /* @__PURE__ */ zt(e), r = !1, s = !1;
		n && (r = !/* @__PURE__ */ Vt(e), s = /* @__PURE__ */ Bt(e), e = rt(e)), i = Array(e.length);
		for (let n = 0, o = e.length; n < o; n++) i[n] = t(r ? s ? Gt(Wt(e[n])) : Wt(e[n]) : e[n], n, void 0, a && a[n]);
	} else if (typeof e == "number") {
		i = Array(e);
		for (let n = 0; n < e; n++) i[n] = t(n + 1, n, void 0, a && a[n]);
	} else if (v(e)) if (e[Symbol.iterator]) i = Array.from(e, (e, n) => t(e, n, void 0, a && a[n]));
	else {
		let n = Object.keys(e);
		i = Array(n.length);
		for (let r = 0, o = n.length; r < o; r++) {
			let o = n[r];
			i[r] = t(e[o], o, r, a && a[r]);
		}
	}
	else i = [];
	return n && (n[r] = i), i;
}
var Rr = (e) => e ? Ea(e) ? Ia(e) : Rr(e.parent) : null, zr = /* @__PURE__ */ s(/* @__PURE__ */ Object.create(null), {
	$: (e) => e,
	$el: (e) => e.vnode.el,
	$data: (e) => e.data,
	$props: (e) => e.props,
	$attrs: (e) => e.attrs,
	$slots: (e) => e.slots,
	$refs: (e) => e.refs,
	$parent: (e) => Rr(e.parent),
	$root: (e) => Rr(e.root),
	$host: (e) => e.ce,
	$emit: (e) => e.emit,
	$options: (e) => Jr(e),
	$forceUpdate: (e) => e.f ||= () => {
		_n(e.update);
	},
	$nextTick: (e) => e.n ||= hn.bind(e.proxy),
	$watch: (e) => Fn.bind(e)
}), Br = (e, n) => e !== t && !e.__isScriptSetup && u(e, n), Vr = {
	get({ _: e }, n) {
		if (n === "__v_skip") return !0;
		let { ctx: r, setupState: i, data: a, props: o, accessCache: s, type: c, appContext: l } = e;
		if (n[0] !== "$") {
			let e = s[n];
			if (e !== void 0) switch (e) {
				case 1: return i[n];
				case 2: return a[n];
				case 4: return r[n];
				case 3: return o[n];
			}
			else if (Br(i, n)) return s[n] = 1, i[n];
			else if (a !== t && u(a, n)) return s[n] = 2, a[n];
			else if (u(o, n)) return s[n] = 3, o[n];
			else if (r !== t && u(r, n)) return s[n] = 4, r[n];
			else Ur && (s[n] = 0);
		}
		let d = zr[n], f, p;
		if (d) return n === "$attrs" && M(e.attrs, "get", ""), d(e);
		if ((f = c.__cssModules) && (f = f[n])) return f;
		if (r !== t && u(r, n)) return s[n] = 4, r[n];
		if (p = l.config.globalProperties, u(p, n)) return p[n];
	},
	set({ _: e }, n, r) {
		let { data: i, setupState: a, ctx: o } = e;
		return Br(a, n) ? (a[n] = r, !0) : i !== t && u(i, n) ? (i[n] = r, !0) : u(e.props, n) || n[0] === "$" && n.slice(1) in e ? !1 : (o[n] = r, !0);
	},
	has({ _: { data: e, setupState: n, accessCache: r, ctx: i, appContext: a, props: o, type: s } }, c) {
		let l;
		return !!(r[c] || e !== t && c[0] !== "$" && u(e, c) || Br(n, c) || u(o, c) || u(i, c) || u(zr, c) || u(a.config.globalProperties, c) || (l = s.__cssModules) && l[c]);
	},
	defineProperty(e, t, n) {
		return n.get == null ? u(n, "value") && this.set(e, t, n.value, null) : e._.accessCache[t] = 0, Reflect.defineProperty(e, t, n);
	}
};
function Hr(e) {
	return d(e) ? e.reduce((e, t) => (e[t] = null, e), {}) : e;
}
var Ur = !0;
function Wr(e) {
	let t = Jr(e), n = e.proxy, i = e.ctx;
	Ur = !1, t.beforeCreate && Kr(t.beforeCreate, e, "bc");
	let { data: a, computed: o, methods: s, watch: c, provide: l, inject: u, created: f, beforeMount: p, mounted: m, beforeUpdate: g, updated: _, activated: y, deactivated: b, beforeDestroy: x, beforeUnmount: S, destroyed: C, unmounted: w, render: ee, renderTracked: te, renderTriggered: ne, errorCaptured: T, serverPrefetch: re, expose: E, inheritAttrs: ie, components: ae, directives: oe, filters: se } = t;
	if (u && Gr(u, i, null), s) for (let e in s) {
		let t = s[e];
		h(t) && (i[e] = t.bind(n));
	}
	if (a) {
		let t = a.call(n, n);
		v(t) && (e.data = /* @__PURE__ */ Ft(t));
	}
	if (Ur = !0, o) for (let e in o) {
		let t = o[e], a = Y({
			get: h(t) ? t.bind(n, n) : h(t.get) ? t.get.bind(n, n) : r,
			set: !h(t) && h(t.set) ? t.set.bind(n) : r
		});
		Object.defineProperty(i, e, {
			enumerable: !0,
			configurable: !0,
			get: () => a.value,
			set: (e) => a.value = e
		});
	}
	if (c) for (let e in c) qr(c[e], i, n, e);
	if (l) {
		let e = h(l) ? l.call(n) : l;
		Reflect.ownKeys(e).forEach((t) => {
			kn(t, e[t]);
		});
	}
	f && Kr(f, e, "c");
	function D(e, t) {
		d(t) ? t.forEach((t) => e(t.bind(n))) : t && e(t.bind(n));
	}
	if (D(Er, p), D(Dr, m), D(Or, g), D(kr, _), D(br, y), D(xr, b), D(Fr, T), D(Pr, te), D(Nr, ne), D(Ar, S), D(jr, w), D(Mr, re), d(E)) if (E.length) {
		let t = e.exposed ||= {};
		E.forEach((e) => {
			Object.defineProperty(t, e, {
				get: () => n[e],
				set: (t) => n[e] = t,
				enumerable: !0
			});
		});
	} else e.exposed ||= {};
	ee && e.render === r && (e.render = ee), ie != null && (e.inheritAttrs = ie), ae && (e.components = ae), oe && (e.directives = oe), re && pr(e);
}
function Gr(e, t, n = r) {
	d(e) && (e = $r(e));
	for (let n in e) {
		let r = e[n], i;
		i = v(r) ? "default" in r ? An(r.from || n, r.default, !0) : An(r.from || n) : An(r), /* @__PURE__ */ P(i) ? Object.defineProperty(t, n, {
			enumerable: !0,
			configurable: !0,
			get: () => i.value,
			set: (e) => i.value = e
		}) : t[n] = i;
	}
}
function Kr(e, t, n) {
	on(d(e) ? e.map((e) => e.bind(t.proxy)) : e.bind(t.proxy), t, n);
}
function qr(e, t, n, r) {
	let i = r.includes(".") ? In(n, r) : () => n[r];
	if (g(e)) {
		let n = t[e];
		h(n) && Nn(i, n);
	} else if (h(e)) Nn(i, e.bind(n));
	else if (v(e)) if (d(e)) e.forEach((e) => qr(e, t, n, r));
	else {
		let r = h(e.handler) ? e.handler.bind(n) : t[e.handler];
		h(r) && Nn(i, r, e);
	}
}
function Jr(e) {
	let t = e.type, { mixins: n, extends: r } = t, { mixins: i, optionsCache: a, config: { optionMergeStrategies: o } } = e.appContext, s = a.get(t), c;
	return s ? c = s : !i.length && !n && !r ? c = t : (c = {}, i.length && i.forEach((e) => Yr(c, e, o, !0)), Yr(c, t, o)), v(t) && a.set(t, c), c;
}
function Yr(e, t, n, r = !1) {
	let { mixins: i, extends: a } = t;
	a && Yr(e, a, n, !0), i && i.forEach((t) => Yr(e, t, n, !0));
	for (let i in t) if (!(r && i === "expose")) {
		let r = Xr[i] || n && n[i];
		e[i] = r ? r(e[i], t[i]) : t[i];
	}
	return e;
}
var Xr = {
	data: Zr,
	props: ti,
	emits: ti,
	methods: ei,
	computed: ei,
	beforeCreate: z,
	created: z,
	beforeMount: z,
	mounted: z,
	beforeUpdate: z,
	updated: z,
	beforeDestroy: z,
	beforeUnmount: z,
	destroyed: z,
	unmounted: z,
	activated: z,
	deactivated: z,
	errorCaptured: z,
	serverPrefetch: z,
	components: ei,
	directives: ei,
	watch: ni,
	provide: Zr,
	inject: Qr
};
function Zr(e, t) {
	return t ? e ? function() {
		return s(h(e) ? e.call(this, this) : e, h(t) ? t.call(this, this) : t);
	} : t : e;
}
function Qr(e, t) {
	return ei($r(e), $r(t));
}
function $r(e) {
	if (d(e)) {
		let t = {};
		for (let n = 0; n < e.length; n++) t[e[n]] = e[n];
		return t;
	}
	return e;
}
function z(e, t) {
	return e ? [...new Set([].concat(e, t))] : t;
}
function ei(e, t) {
	return e ? s(/* @__PURE__ */ Object.create(null), e, t) : t;
}
function ti(e, t) {
	return e ? d(e) && d(t) ? [.../* @__PURE__ */ new Set([...e, ...t])] : s(/* @__PURE__ */ Object.create(null), Hr(e), Hr(t ?? {})) : t;
}
function ni(e, t) {
	if (!e) return t;
	if (!t) return e;
	let n = s(/* @__PURE__ */ Object.create(null), e);
	for (let r in t) n[r] = z(e[r], t[r]);
	return n;
}
function ri() {
	return {
		app: null,
		config: {
			isNativeTag: i,
			performance: !1,
			globalProperties: {},
			optionMergeStrategies: {},
			errorHandler: void 0,
			warnHandler: void 0,
			compilerOptions: {}
		},
		mixins: [],
		components: {},
		directives: {},
		provides: /* @__PURE__ */ Object.create(null),
		optionsCache: /* @__PURE__ */ new WeakMap(),
		propsCache: /* @__PURE__ */ new WeakMap(),
		emitsCache: /* @__PURE__ */ new WeakMap()
	};
}
var ii = 0;
function ai(e, t) {
	return function(n, r = null) {
		h(n) || (n = s({}, n)), r != null && !v(r) && (r = null);
		let i = ri(), a = /* @__PURE__ */ new WeakSet(), o = [], c = !1, l = i.app = {
			_uid: ii++,
			_component: n,
			_props: r,
			_container: null,
			_context: i,
			_instance: null,
			version: za,
			get config() {
				return i.config;
			},
			set config(e) {},
			use(e, ...t) {
				return a.has(e) || (e && h(e.install) ? (a.add(e), e.install(l, ...t)) : h(e) && (a.add(e), e(l, ...t))), l;
			},
			mixin(e) {
				return i.mixins.includes(e) || i.mixins.push(e), l;
			},
			component(e, t) {
				return t ? (i.components[e] = t, l) : i.components[e];
			},
			directive(e, t) {
				return t ? (i.directives[e] = t, l) : i.directives[e];
			},
			mount(a, o, s) {
				if (!c) {
					let u = l._ceVNode || K(n, r);
					return u.appContext = i, s === !0 ? s = "svg" : s === !1 && (s = void 0), o && t ? t(u, a) : e(u, a, s), c = !0, l._container = a, a.__vue_app__ = l, Ia(u.component);
				}
			},
			onUnmount(e) {
				o.push(e);
			},
			unmount() {
				c && (on(o, l._instance, 16), e(null, l._container), delete l._container.__vue_app__);
			},
			provide(e, t) {
				return i.provides[e] = t, l;
			},
			runWithContext(e) {
				let t = oi;
				oi = l;
				try {
					return e();
				} finally {
					oi = t;
				}
			}
		};
		return l;
	};
}
var oi = null, si = (e, t) => t === "modelValue" || t === "model-value" ? e.modelModifiers : e[`${t}Modifiers`] || e[`${T(t)}Modifiers`] || e[`${E(t)}Modifiers`];
function ci(e, n, ...r) {
	if (e.isUnmounted) return;
	let i = e.vnode.props || t, a = r, o = n.startsWith("update:"), s = o && si(i, n.slice(7));
	s && (s.trim && (a = r.map((e) => g(e) ? e.trim() : e)), s.number && (a = r.map(ce)));
	let c, l = i[c = ae(n)] || i[c = ae(T(n))];
	!l && o && (l = i[c = ae(E(n))]), l && on(l, e, 6, a);
	let u = i[c + "Once"];
	if (u) {
		if (!e.emitted) e.emitted = {};
		else if (e.emitted[c]) return;
		e.emitted[c] = !0, on(u, e, 6, a);
	}
}
var li = /* @__PURE__ */ new WeakMap();
function ui(e, t, n = !1) {
	let r = n ? li : t.emitsCache, i = r.get(e);
	if (i !== void 0) return i;
	let a = e.emits, o = {}, c = !1;
	if (!h(e)) {
		let r = (e) => {
			let n = ui(e, t, !0);
			n && (c = !0, s(o, n));
		};
		!n && t.mixins.length && t.mixins.forEach(r), e.extends && r(e.extends), e.mixins && e.mixins.forEach(r);
	}
	return !a && !c ? (v(e) && r.set(e, null), null) : (d(a) ? a.forEach((e) => o[e] = null) : s(o, a), v(e) && r.set(e, o), o);
}
function di(e, t) {
	return !e || !a(t) ? !1 : (t = t.slice(2), t = t === "Once" ? t : t.replace(/Once$/, ""), u(e, t[0].toLowerCase() + t.slice(1)) || u(e, E(t)) || u(e, t));
}
function fi(e) {
	let { type: t, vnode: n, proxy: r, withProxy: i, propsOptions: [a], slots: s, attrs: c, emit: l, render: u, renderCache: d, props: f, data: p, setupState: m, ctx: h, inheritAttrs: g } = e, _ = En(e), v, y;
	try {
		if (n.shapeFlag & 4) {
			let e = i || r, t = e;
			v = pa(u.call(t, e, d, f, m, p, h)), y = c;
		} else {
			let e = t;
			v = pa(e.length > 1 ? e(f, {
				attrs: c,
				slots: s,
				emit: l
			}) : e(f, null)), y = t.props ? c : pi(c);
		}
	} catch (t) {
		Zi.length = 0, sn(t, e, 1), v = K(H);
	}
	let b = v;
	if (y && g !== !1) {
		let e = Object.keys(y), { shapeFlag: t } = b;
		e.length && t & 7 && (a && e.some(o) && (y = mi(y, a)), b = ua(b, y, !1, !0));
	}
	return n.dirs && (b = ua(b, null, !1, !0), b.dirs = b.dirs ? b.dirs.concat(n.dirs) : n.dirs), n.transition && ur(b, n.transition), v = b, En(_), v;
}
var pi = (e) => {
	let t;
	for (let n in e) (n === "class" || n === "style" || a(n)) && ((t ||= {})[n] = e[n]);
	return t;
}, mi = (e, t) => {
	let n = {};
	for (let r in e) (!o(r) || !(r.slice(9) in t)) && (n[r] = e[r]);
	return n;
};
function hi(e, t, n) {
	let { props: r, children: i, component: a } = e, { props: o, children: s, patchFlag: c } = t, l = a.emitsOptions;
	if (t.dirs || t.transition) return !0;
	if (n && c >= 0) {
		if (c & 1024) return !0;
		if (c & 16) return r ? gi(r, o, l) : !!o;
		if (c & 8) {
			let e = t.dynamicProps;
			for (let t = 0; t < e.length; t++) {
				let n = e[t];
				if (_i(o, r, n) && !di(l, n)) return !0;
			}
		}
	} else return (i || s) && (!s || !s.$stable) ? !0 : r === o ? !1 : r ? !o || gi(r, o, l) : !!o;
	return !1;
}
function gi(e, t, n) {
	let r = Object.keys(t);
	if (r.length !== Object.keys(e).length) return !0;
	for (let i = 0; i < r.length; i++) {
		let a = r[i];
		if (_i(t, e, a) && !di(n, a)) return !0;
	}
	return !1;
}
function _i(e, t, n) {
	let r = e[n], i = t[n];
	return n === "style" && v(r) && v(i) ? !xe(r, i) : r !== i;
}
function vi({ vnode: e, parent: t, suspense: n }, r) {
	for (; t;) {
		let n = t.subTree;
		if (n.suspense && n.suspense.activeBranch === e && (n.suspense.vnode.el = n.el = r, e = n), n === e) (e = t.vnode).el = r, t = t.parent;
		else break;
	}
	n && n.activeBranch === e && (n.vnode.el = r);
}
var yi = {}, bi = () => Object.create(yi), xi = (e) => Object.getPrototypeOf(e) === yi;
function Si(e, t, n, r = !1) {
	let i = {}, a = bi();
	e.propsDefaults = /* @__PURE__ */ Object.create(null), wi(e, t, i, a);
	for (let t in e.propsOptions[0]) t in i || (i[t] = void 0);
	e.props = n ? r ? i : /* @__PURE__ */ It(i) : e.type.props ? i : a, e.attrs = a;
}
function Ci(e, t, n, r) {
	let { props: i, attrs: a, vnode: { patchFlag: o } } = e, s = /* @__PURE__ */ N(i), [c] = e.propsOptions, l = !1;
	if ((r || o > 0) && !(o & 16)) {
		if (o & 8) {
			let n = e.vnode.dynamicProps;
			for (let r = 0; r < n.length; r++) {
				let o = n[r];
				if (di(e.emitsOptions, o)) continue;
				let d = t[o];
				if (c) if (u(a, o)) d !== a[o] && (a[o] = d, l = !0);
				else {
					let t = T(o);
					i[t] = Ti(c, s, t, d, e, !1);
				}
				else d !== a[o] && (a[o] = d, l = !0);
			}
		}
	} else {
		wi(e, t, i, a) && (l = !0);
		let r;
		for (let a in s) (!t || !u(t, a) && ((r = E(a)) === a || !u(t, r))) && (c ? n && (n[a] !== void 0 || n[r] !== void 0) && (i[a] = Ti(c, s, a, void 0, e, !0)) : delete i[a]);
		if (a !== s) for (let e in a) (!t || !u(t, e)) && (delete a[e], l = !0);
	}
	l && tt(e.attrs, "set", "");
}
function wi(e, n, r, i) {
	let [a, o] = e.propsOptions, s = !1, c;
	if (n) for (let t in n) {
		if (ee(t)) continue;
		let l = n[t], d;
		a && u(a, d = T(t)) ? !o || !o.includes(d) ? r[d] = l : (c ||= {})[d] = l : di(e.emitsOptions, t) || (!(t in i) || l !== i[t]) && (i[t] = l, s = !0);
	}
	if (o) {
		let n = /* @__PURE__ */ N(r), i = c || t;
		for (let t = 0; t < o.length; t++) {
			let s = o[t];
			r[s] = Ti(a, n, s, i[s], e, !u(i, s));
		}
	}
	return s;
}
function Ti(e, t, n, r, i, a) {
	let o = e[n];
	if (o != null) {
		let e = u(o, "default");
		if (e && r === void 0) {
			let e = o.default;
			if (o.type !== Function && !o.skipFactory && h(e)) {
				let { propsDefaults: a } = i;
				if (n in a) r = a[n];
				else {
					let o = wa(i);
					r = a[n] = e.call(null, t), o();
				}
			} else r = e;
			i.ce && i.ce._setProp(n, r);
		}
		o[0] && (a && !e ? r = !1 : o[1] && (r === "" || r === E(n)) && (r = !0));
	}
	return r;
}
var Ei = /* @__PURE__ */ new WeakMap();
function Di(e, r, i = !1) {
	let a = i ? Ei : r.propsCache, o = a.get(e);
	if (o) return o;
	let c = e.props, l = {}, f = [], p = !1;
	if (!h(e)) {
		let t = (e) => {
			p = !0;
			let [t, n] = Di(e, r, !0);
			s(l, t), n && f.push(...n);
		};
		!i && r.mixins.length && r.mixins.forEach(t), e.extends && t(e.extends), e.mixins && e.mixins.forEach(t);
	}
	if (!c && !p) return v(e) && a.set(e, n), n;
	if (d(c)) for (let e = 0; e < c.length; e++) {
		let n = T(c[e]);
		Oi(n) && (l[n] = t);
	}
	else if (c) for (let e in c) {
		let t = T(e);
		if (Oi(t)) {
			let n = c[e], r = l[t] = d(n) || h(n) ? { type: n } : s({}, n), i = r.type, a = !1, o = !0;
			if (d(i)) for (let e = 0; e < i.length; ++e) {
				let t = i[e], n = h(t) && t.name;
				if (n === "Boolean") {
					a = !0;
					break;
				}
				n === "String" && (o = !1);
			}
			else a = h(i) && i.name === "Boolean";
			r[0] = a, r[1] = o, (a || u(r, "default")) && f.push(t);
		}
	}
	let m = [l, f];
	return v(e) && a.set(e, m), m;
}
function Oi(e) {
	return e[0] !== "$" && !ee(e);
}
var ki = (e) => e === "_" || e === "_ctx" || e === "$stable", Ai = (e) => d(e) ? e.map(pa) : [pa(e)], ji = (e, t, n) => {
	if (t._n) return t;
	let r = Dn((...e) => Ai(t(...e)), n);
	return r._c = !1, r;
}, Mi = (e, t, n) => {
	let r = e._ctx;
	for (let n in e) {
		if (ki(n)) continue;
		let i = e[n];
		if (h(i)) t[n] = ji(n, i, r);
		else if (i != null) {
			let e = Ai(i);
			t[n] = () => e;
		}
	}
}, Ni = (e, t) => {
	let n = Ai(t);
	e.slots.default = () => n;
}, Pi = (e, t, n) => {
	for (let r in t) (n || !ki(r)) && (e[r] = t[r]);
}, Fi = (e, t, n) => {
	let r = e.slots = bi();
	if (e.vnode.shapeFlag & 32) {
		let e = t._;
		e ? (Pi(r, t, n), n && D(r, "_", e, !0)) : Mi(t, r);
	} else t && Ni(e, t);
}, Ii = (e, n, r) => {
	let { vnode: i, slots: a } = e, o = !0, s = t;
	if (i.shapeFlag & 32) {
		let e = n._;
		e ? r && e === 1 ? o = !1 : Pi(a, n, r) : (o = !n.$stable, Mi(n, a)), s = n;
	} else n && (Ni(e, n), s = { default: 1 });
	if (o) for (let e in a) !ki(e) && s[e] == null && delete a[e];
}, B = Ji;
function Li(e) {
	return Ri(e);
}
function Ri(e, i) {
	let a = de();
	a.__VUE__ = !0;
	let { insert: o, remove: s, patchProp: c, createElement: l, createText: u, createComment: d, setText: f, setElementText: p, parentNode: m, nextSibling: h, setScopeId: g = r, insertStaticContent: _ } = e, v = (e, t, n, r = null, i = null, a = null, o = void 0, s = null, c = !!t.dynamicChildren) => {
		if (e === t) return;
		e && !aa(e, t) && (r = be(e), ge(e, i, a, !0), e = null), t.patchFlag === -2 && (c = !1, t.dynamicChildren = null);
		let { type: l, ref: u, shapeFlag: d } = t;
		switch (l) {
			case Yi:
				y(e, t, n, r);
				break;
			case H:
				b(e, t, n, r);
				break;
			case Xi:
				e ?? x(t, n, r, o);
				break;
			case V:
				ae(e, t, n, r, i, a, o, s, c);
				break;
			default: d & 1 ? w(e, t, n, r, i, a, o, s, c) : d & 6 ? oe(e, t, n, r, i, a, o, s, c) : (d & 64 || d & 128) && l.process(e, t, n, r, i, a, o, s, c, Ce);
		}
		u != null && i ? gr(u, e && e.ref, a, t || e, !t) : u == null && e && e.ref != null && gr(e.ref, null, a, e, !0);
	}, y = (e, t, n, r) => {
		if (e == null) o(t.el = u(t.children), n, r);
		else {
			let n = t.el = e.el;
			t.children !== e.children && f(n, t.children);
		}
	}, b = (e, t, n, r) => {
		e == null ? o(t.el = d(t.children || ""), n, r) : t.el = e.el;
	}, x = (e, t, n, r) => {
		[e.el, e.anchor] = _(e.children, t, n, r, e.el, e.anchor);
	}, S = ({ el: e, anchor: t }, n, r) => {
		let i;
		for (; e && e !== t;) i = h(e), o(e, n, r), e = i;
		o(t, n, r);
	}, C = ({ el: e, anchor: t }) => {
		let n;
		for (; e && e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, w = (e, t, n, r, i, a, o, s, c) => {
		if (t.type === "svg" ? o = "svg" : t.type === "math" && (o = "mathml"), e == null) te(t, n, r, i, a, o, s, c);
		else {
			let n = e.el && e.el._isVueCE ? e.el : null;
			try {
				n && n._beginPatch(), re(e, t, i, a, o, s, c);
			} finally {
				n && n._endPatch();
			}
		}
	}, te = (e, t, n, r, i, a, s, u) => {
		let d, f, { props: m, shapeFlag: h, transition: g, dirs: _ } = e;
		if (d = e.el = l(e.type, a, m && m.is, m), h & 8 ? p(d, e.children) : h & 16 && T(e.children, d, null, r, i, zi(e, a), s, u), _ && On(e, null, r, "created"), ne(d, e, e.scopeId, s, r), m) {
			for (let e in m) e !== "value" && !ee(e) && c(d, e, null, m[e], a, r);
			"value" in m && c(d, "value", null, m.value, a), (f = m.onVnodeBeforeMount) && _a(f, r, e);
		}
		_ && On(e, null, r, "beforeMount");
		let v = Vi(i, g);
		v && g.beforeEnter(d), o(d, t, n), ((f = m && m.onVnodeMounted) || v || _) && B(() => {
			try {
				f && _a(f, r, e), v && g.enter(d), _ && On(e, null, r, "mounted");
			} finally {}
		}, i);
	}, ne = (e, t, n, r, i) => {
		if (n && g(e, n), r) for (let t = 0; t < r.length; t++) g(e, r[t]);
		if (i) {
			let n = i.subTree;
			if (t === n || qi(n.type) && (n.ssContent === t || n.ssFallback === t)) {
				let t = i.vnode;
				ne(e, t, t.scopeId, t.slotScopeIds, i.parent);
			}
		}
	}, T = (e, t, n, r, i, a, o, s, c = 0) => {
		for (let l = c; l < e.length; l++) {
			let c = e[l] = s ? ma(e[l]) : pa(e[l]);
			v(null, c, t, n, r, i, a, o, s);
		}
	}, re = (e, n, r, i, a, o, s) => {
		let l = n.el = e.el, { patchFlag: u, dynamicChildren: d, dirs: f } = n;
		u |= e.patchFlag & 16;
		let m = e.props || t, h = n.props || t, g;
		if (r && Bi(r, !1), (g = h.onVnodeBeforeUpdate) && _a(g, r, n, e), f && On(n, e, r, "beforeUpdate"), r && Bi(r, !0), d && (!e.dynamicChildren || e.dynamicChildren.length !== d.length) && (u = 0, s = !1, d = null), (m.innerHTML && h.innerHTML == null || m.textContent && h.textContent == null) && p(l, ""), d ? E(e.dynamicChildren, d, l, r, i, zi(n, a), o) : s || fe(e, n, l, null, r, i, zi(n, a), o, !1), u > 0) {
			if (u & 16) ie(l, m, h, r, a);
			else if (u & 2 && m.class !== h.class && c(l, "class", null, h.class, a), u & 4 && c(l, "style", m.style, h.style, a), u & 8) {
				let e = n.dynamicProps;
				for (let t = 0; t < e.length; t++) {
					let n = e[t], i = m[n], o = h[n];
					(o !== i || n === "value") && c(l, n, i, o, a, r);
				}
			}
			u & 1 && e.children !== n.children && p(l, n.children);
		} else !s && d == null && ie(l, m, h, r, a);
		((g = h.onVnodeUpdated) || f) && B(() => {
			g && _a(g, r, n, e), f && On(n, e, r, "updated");
		}, i);
	}, E = (e, t, n, r, i, a, o) => {
		for (let s = 0; s < t.length; s++) {
			let c = e[s], l = t[s], u = c.el && (c.type === V || !aa(c, l) || c.shapeFlag & 198) ? m(c.el) : n;
			v(c, l, u, null, r, i, a, o, !0);
		}
	}, ie = (e, n, r, i, a) => {
		if (n !== r) {
			if (n !== t) for (let t in n) !ee(t) && !(t in r) && c(e, t, n[t], null, a, i);
			for (let t in r) {
				if (ee(t)) continue;
				let o = r[t], s = n[t];
				o !== s && t !== "value" && c(e, t, s, o, a, i);
			}
			"value" in r && c(e, "value", n.value, r.value, a);
		}
	}, ae = (e, t, n, r, i, a, s, c, l) => {
		let d = t.el = e ? e.el : u(""), f = t.anchor = e ? e.anchor : u(""), { patchFlag: p, dynamicChildren: m, slotScopeIds: h } = t;
		h && (c = c ? c.concat(h) : h), e == null ? (o(d, n, r), o(f, n, r), T(t.children || [], n, f, i, a, s, c, l)) : p > 0 && p & 64 && m && e.dynamicChildren && e.dynamicChildren.length === m.length ? (E(e.dynamicChildren, m, n, i, a, s, c), (t.key != null || i && t === i.subTree) && Hi(e, t, !0)) : fe(e, t, n, f, i, a, s, c, l);
	}, oe = (e, t, n, r, i, a, o, s, c) => {
		t.slotScopeIds = s, e == null ? t.shapeFlag & 512 ? i.ctx.activate(t, n, r, o, c) : D(t, n, r, i, a, o, c) : ce(e, t, c);
	}, D = (e, t, n, r, i, a, o) => {
		let s = e.component = ba(e, r, i);
		if (yr(e) && (s.ctx.renderer = Ce), Oa(s, !1, o), s.asyncDep) {
			if (i && i.registerDep(s, le, o), !e.el) {
				let r = s.subTree = K(H);
				b(null, r, t, n), e.placeholder = r.el;
			}
		} else le(s, e, t, n, i, a, o);
	}, ce = (e, t, n) => {
		let r = t.component = e.component;
		if (hi(e, t, n)) if (r.asyncDep && !r.asyncResolved) {
			ue(r, t, n);
			return;
		} else r.next = t, r.update();
		else t.el = e.el, r.vnode = t;
	}, le = (e, t, n, r, i, a, o) => {
		let s = () => {
			if (e.isMounted) {
				let { next: t, bu: n, u: r, parent: s, vnode: c } = e;
				{
					let n = Wi(e);
					if (n) {
						t && (t.el = c.el, ue(e, t, o)), n.asyncDep.then(() => {
							B(() => {
								e.isUnmounted || l();
							}, i);
						});
						return;
					}
				}
				let u = t, d;
				Bi(e, !1), t ? (t.el = c.el, ue(e, t, o)) : t = c, n && se(n), (d = t.props && t.props.onVnodeBeforeUpdate) && _a(d, s, t, c), Bi(e, !0);
				let f = fi(e), p = e.subTree;
				e.subTree = f, v(p, f, m(p.el), be(p), e, i, a), t.el = f.el, u === null && vi(e, f.el), r && B(r, i), (d = t.props && t.props.onVnodeUpdated) && B(() => _a(d, s, t, c), i);
			} else {
				let o, { el: s, props: c } = t, { bm: l, m: u, parent: d, root: f, type: p } = e, m = vr(t);
				if (Bi(e, !1), l && se(l), !m && (o = c && c.onVnodeBeforeMount) && _a(o, d, t), Bi(e, !0), s && we) {
					let t = () => {
						e.subTree = fi(e), we(s, e.subTree, e, i, null);
					};
					m && p.__asyncHydrate ? p.__asyncHydrate(s, e, t) : t();
				} else {
					f.ce && f.ce._hasShadowRoot() && f.ce._injectChildStyle(p, e.parent ? e.parent.type : void 0);
					let o = e.subTree = fi(e);
					v(null, o, n, r, e, i, a), t.el = o.el;
				}
				if (u && B(u, i), !m && (o = c && c.onVnodeMounted)) {
					let e = t;
					B(() => _a(o, d, e), i);
				}
				(t.shapeFlag & 256 || d && vr(d.vnode) && d.vnode.shapeFlag & 256) && e.a && B(e.a, i), e.isMounted = !0, t = n = r = null;
			}
		};
		e.scope.on();
		let c = e.effect = new ke(s);
		e.scope.off();
		let l = e.update = c.run.bind(c), u = e.job = c.runIfDirty.bind(c);
		u.i = e, u.id = e.uid, c.scheduler = () => _n(u), Bi(e, !0), l();
	}, ue = (e, t, n) => {
		t.component = e;
		let r = e.vnode.props;
		e.vnode = t, e.next = null, Ci(e, t.props, r, n), Ii(e, t.children, n), We(), bn(e), Ge();
	}, fe = (e, t, n, r, i, a, o, s, c = !1) => {
		let l = e && e.children, u = e ? e.shapeFlag : 0, d = t.children, { patchFlag: f, shapeFlag: m } = t;
		if (f > 0) {
			if (f & 128) {
				me(l, d, n, r, i, a, o, s, c);
				return;
			}
			if (f & 256) {
				pe(l, d, n, r, i, a, o, s, c);
				return;
			}
		}
		m & 8 ? (u & 16 && ye(l, i, a), d !== l && p(n, d)) : u & 16 ? m & 16 ? me(l, d, n, r, i, a, o, s, c) : ye(l, i, a, !0) : (u & 8 && p(n, ""), m & 16 && T(d, n, r, i, a, o, s, c));
	}, pe = (e, t, r, i, a, o, s, c, l) => {
		e ||= n, t ||= n;
		let u = e.length, d = t.length, f = Math.min(u, d), p;
		for (p = 0; p < f; p++) {
			let n = t[p] = l ? ma(t[p]) : pa(t[p]);
			v(e[p], n, r, null, a, o, s, c, l);
		}
		u > d ? ye(e, a, o, !0, !1, f) : T(t, r, i, a, o, s, c, l, f);
	}, me = (e, t, r, i, a, o, s, c, l) => {
		let u = 0, d = t.length, f = e.length - 1, p = d - 1;
		for (; u <= f && u <= p;) {
			let n = e[u], i = t[u] = l ? ma(t[u]) : pa(t[u]);
			if (aa(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			u++;
		}
		for (; u <= f && u <= p;) {
			let n = e[f], i = t[p] = l ? ma(t[p]) : pa(t[p]);
			if (aa(n, i)) v(n, i, r, null, a, o, s, c, l);
			else break;
			f--, p--;
		}
		if (u > f) {
			if (u <= p) {
				let e = p + 1, n = e < d ? t[e].el : i;
				for (; u <= p;) v(null, t[u] = l ? ma(t[u]) : pa(t[u]), r, n, a, o, s, c, l), u++;
			}
		} else if (u > p) for (; u <= f;) ge(e[u], a, o, !0), u++;
		else {
			let m = u, h = u, g = /* @__PURE__ */ new Map();
			for (u = h; u <= p; u++) {
				let e = t[u] = l ? ma(t[u]) : pa(t[u]);
				e.key != null && g.set(e.key, u);
			}
			let _, y = 0, b = p - h + 1, x = !1, S = 0, C = Array(b);
			for (u = 0; u < b; u++) C[u] = 0;
			for (u = m; u <= f; u++) {
				let n = e[u];
				if (y >= b) {
					ge(n, a, o, !0);
					continue;
				}
				let i;
				if (n.key != null) i = g.get(n.key);
				else for (_ = h; _ <= p; _++) if (C[_ - h] === 0 && aa(n, t[_])) {
					i = _;
					break;
				}
				i === void 0 ? ge(n, a, o, !0) : (C[i - h] = u + 1, i >= S ? S = i : x = !0, v(n, t[i], r, null, a, o, s, c, l), y++);
			}
			let w = x ? Ui(C) : n;
			for (_ = w.length - 1, u = b - 1; u >= 0; u--) {
				let e = h + u, n = t[e], f = t[e + 1], p = e + 1 < d ? f.el || Ki(f) : i;
				C[u] === 0 ? v(null, n, r, p, a, o, s, c, l) : x && (_ < 0 || u !== w[_] ? he(n, r, p, 2) : _--);
			}
		}
	}, he = (e, t, n, r, i = null) => {
		let { el: a, type: c, transition: l, children: u, shapeFlag: d } = e;
		if (d & 6) {
			he(e.component.subTree, t, n, r);
			return;
		}
		if (d & 128) {
			e.suspense.move(t, n, r);
			return;
		}
		if (d & 64) {
			c.move(e, t, n, Ce);
			return;
		}
		if (c === V) {
			o(a, t, n);
			for (let e = 0; e < u.length; e++) he(u[e], t, n, r);
			o(e.anchor, t, n);
			return;
		}
		if (c === Xi) {
			S(e, t, n);
			return;
		}
		if (r !== 2 && d & 1 && l) if (r === 0) l.persisted && !a[Zn] ? o(a, t, n) : (l.beforeEnter(a), o(a, t, n), B(() => l.enter(a), i));
		else {
			let { leave: r, delayLeave: i, afterLeave: c } = l, u = () => {
				e.ctx.isUnmounted ? s(a) : o(a, t, n);
			}, d = () => {
				let e = a._isLeaving || !!a[Zn];
				a._isLeaving && a[Zn](!0), l.persisted && !e ? u() : r(a, () => {
					u(), c && c();
				});
			};
			i ? i(a, u, d) : d();
		}
		else o(a, t, n);
	}, ge = (e, t, n, r = !1, i = !1) => {
		let { type: a, props: o, ref: s, children: c, dynamicChildren: l, shapeFlag: u, patchFlag: d, dirs: f, cacheIndex: p, memo: m } = e;
		if (d === -2 && (i = !1), s != null && (We(), gr(s, null, n, e, !0), Ge()), p != null && (t.renderCache[p] = void 0), u & 256) {
			t.ctx.deactivate(e);
			return;
		}
		let h = u & 1 && f, g = !vr(e), _;
		if (g && (_ = o && o.onVnodeBeforeUnmount) && _a(_, t, e), u & 6) ve(e.component, n, r);
		else {
			if (u & 128) {
				e.suspense.unmount(n, r);
				return;
			}
			h && On(e, null, t, "beforeUnmount"), u & 64 ? e.type.remove(e, t, n, Ce, r) : l && !l.hasOnce && (a !== V || d > 0 && d & 64) ? ye(l, t, n, !1, !0) : (a === V && d & 384 || !i && u & 16) && ye(c, t, n), r && O(e);
		}
		let v = m != null && p == null;
		(g && (_ = o && o.onVnodeUnmounted) || h || v) && B(() => {
			_ && _a(_, t, e), h && On(e, null, t, "unmounted"), v && (e.el = null);
		}, n);
	}, O = (e) => {
		let { type: t, el: n, anchor: r, transition: i } = e;
		if (t === V) {
			_e(n, r);
			return;
		}
		if (t === Xi) {
			C(e);
			return;
		}
		let a = () => {
			s(n), i && !i.persisted && i.afterLeave && i.afterLeave();
		};
		if (e.shapeFlag & 1 && i && !i.persisted) {
			let { leave: t, delayLeave: r } = i, o = () => t(n, a);
			r ? r(e.el, a, o) : o();
		} else a();
	}, _e = (e, t) => {
		let n;
		for (; e !== t;) n = h(e), s(e), e = n;
		s(t);
	}, ve = (e, t, n) => {
		let { bum: r, scope: i, job: a, subTree: o, um: s, m: c, a: l } = e;
		Gi(c), Gi(l), r && se(r), i.stop(), a && (a.flags |= 8, ge(o, e, t, n)), s && B(s, t), B(() => {
			e.isUnmounted = !0;
		}, t);
	}, ye = (e, t, n, r = !1, i = !1, a = 0) => {
		for (let o = a; o < e.length; o++) ge(e[o], t, n, r, i);
	}, be = (e) => {
		if (e.shapeFlag & 6) return be(e.component.subTree);
		if (e.shapeFlag & 128) return e.suspense.next();
		let t = h(e.anchor || e.el), n = t && t[Rn];
		return n ? h(n) : t;
	}, xe = !1, Se = (e, t, n) => {
		let r;
		e == null ? t._vnode && (ge(t._vnode, null, null, !0), r = t._vnode.component) : v(t._vnode || null, e, t, null, null, null, n), t._vnode = e, xe ||= (xe = !0, bn(r), xn(), !1);
	}, Ce = {
		p: v,
		um: ge,
		m: he,
		r: O,
		mt: D,
		mc: T,
		pc: fe,
		pbc: E,
		n: be,
		o: e
	}, k, we;
	return i && ([k, we] = i(Ce)), {
		render: Se,
		hydrate: k,
		createApp: ai(Se, k)
	};
}
function zi({ type: e, props: t }, n) {
	return n === "svg" && e === "foreignObject" || n === "mathml" && e === "annotation-xml" && t && t.encoding && t.encoding.includes("html") ? void 0 : n;
}
function Bi({ effect: e, job: t }, n) {
	n ? (e.flags |= 32, t.flags |= 4) : (e.flags &= -33, t.flags &= -5);
}
function Vi(e, t) {
	return (!e || e && !e.pendingBranch) && t && !t.persisted;
}
function Hi(e, t, n = !1) {
	let r = e.children, i = t.children;
	if (d(r) && d(i)) for (let e = 0; e < r.length; e++) {
		let t = r[e], a = i[e];
		a.shapeFlag & 1 && !a.dynamicChildren && ((a.patchFlag <= 0 || a.patchFlag === 32) && (a = i[e] = ma(i[e]), a.el = t.el), !n && a.patchFlag !== -2 && Hi(t, a)), a.type === Yi && (a.patchFlag === -1 && (a = i[e] = ma(a)), a.el = t.el), a.type === H && !a.el && (a.el = t.el);
	}
}
function Ui(e) {
	let t = e.slice(), n = [0], r, i, a, o, s, c = e.length;
	for (r = 0; r < c; r++) {
		let c = e[r];
		if (c !== 0) {
			if (i = n[n.length - 1], e[i] < c) {
				t[r] = i, n.push(r);
				continue;
			}
			for (a = 0, o = n.length - 1; a < o;) s = a + o >> 1, e[n[s]] < c ? a = s + 1 : o = s;
			c < e[n[a]] && (a > 0 && (t[r] = n[a - 1]), n[a] = r);
		}
	}
	for (a = n.length, o = n[a - 1]; a-- > 0;) n[a] = o, o = t[o];
	return n;
}
function Wi(e) {
	let t = e.subTree.component;
	if (t) return t.asyncDep && !t.asyncResolved ? t : Wi(t);
}
function Gi(e) {
	if (e) for (let t = 0; t < e.length; t++) e[t].flags |= 8;
}
function Ki(e) {
	if (e.placeholder) return e.placeholder;
	let t = e.component;
	return t ? Ki(t.subTree) : null;
}
var qi = (e) => e.__isSuspense;
function Ji(e, t) {
	t && t.pendingBranch ? d(e) ? t.effects.push(...e) : t.effects.push(e) : yn(e);
}
var V = /* @__PURE__ */ Symbol.for("v-fgt"), Yi = /* @__PURE__ */ Symbol.for("v-txt"), H = /* @__PURE__ */ Symbol.for("v-cmt"), Xi = /* @__PURE__ */ Symbol.for("v-stc"), Zi = [], Qi = null;
function U(e = !1) {
	Zi.push(Qi = e ? null : []);
}
function $i() {
	Zi.pop(), Qi = Zi[Zi.length - 1] || null;
}
var ea = 1;
function ta(e, t = !1) {
	ea += e, e < 0 && Qi && t && (Qi.hasOnce = !0);
}
function na(e) {
	return e.dynamicChildren = ea > 0 ? Qi || n : null, $i(), ea > 0 && Qi && Qi.push(e), e;
}
function W(e, t, n, r, i, a) {
	return na(G(e, t, n, r, i, a, !0));
}
function ra(e, t, n, r, i) {
	return na(K(e, t, n, r, i, !0));
}
function ia(e) {
	return e ? e.__v_isVNode === !0 : !1;
}
function aa(e, t) {
	return e.type === t.type && e.key === t.key;
}
var oa = ({ key: e }) => e ?? null, sa = ({ ref: e, ref_key: t, ref_for: n }) => (typeof e == "number" && (e = "" + e), e == null ? null : g(e) || /* @__PURE__ */ P(e) || h(e) ? {
	i: wn,
	r: e,
	k: t,
	f: !!n
} : e);
function G(e, t = null, n = null, r = 0, i = null, a = e === V ? 0 : 1, o = !1, s = !1) {
	let c = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e,
		props: t,
		key: t && oa(t),
		ref: t && sa(t),
		scopeId: Tn,
		slotScopeIds: null,
		children: n,
		component: null,
		suspense: null,
		ssContent: null,
		ssFallback: null,
		dirs: null,
		transition: null,
		el: null,
		anchor: null,
		target: null,
		targetStart: null,
		targetAnchor: null,
		staticCount: 0,
		shapeFlag: a,
		patchFlag: r,
		dynamicProps: i,
		dynamicChildren: null,
		appContext: null,
		ctx: wn
	};
	return s ? (ha(c, n), a & 128 && e.normalize(c)) : n && (c.shapeFlag |= g(n) ? 8 : 16), ea > 0 && !o && Qi && (c.patchFlag > 0 || a & 6) && c.patchFlag !== 32 && Qi.push(c), c;
}
var K = ca;
function ca(e, t = null, n = null, r = 0, i = null, a = !1) {
	if ((!e || e === Ir) && (e = H), ia(e)) {
		let r = ua(e, t, !0);
		return n && ha(r, n), ea > 0 && !a && Qi && (r.shapeFlag & 6 ? Qi[Qi.indexOf(e)] = r : Qi.push(r)), r.patchFlag = -2, r;
	}
	if (La(e) && (e = e.__vccOpts), t) {
		t = la(t);
		let { class: e, style: n } = t;
		e && !g(e) && (t.class = O(e)), v(n) && (/* @__PURE__ */ Ht(n) && !d(n) && (n = s({}, n)), t.style = fe(n));
	}
	let o = g(e) ? 1 : qi(e) ? 128 : zn(e) ? 64 : v(e) ? 4 : h(e) ? 2 : 0;
	return G(e, t, n, r, i, o, a, !0);
}
function la(e) {
	return e ? /* @__PURE__ */ Ht(e) || xi(e) ? s({}, e) : e : null;
}
function ua(e, t, n = !1, r = !1) {
	let { props: i, ref: a, patchFlag: o, children: s, transition: c } = e, l = t ? ga(i || {}, t) : i, u = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: e.type,
		props: l,
		key: l && oa(l),
		ref: t && t.ref ? n && a ? d(a) ? a.concat(sa(t)) : [a, sa(t)] : sa(t) : a,
		scopeId: e.scopeId,
		slotScopeIds: e.slotScopeIds,
		children: s,
		target: e.target,
		targetStart: e.targetStart,
		targetAnchor: e.targetAnchor,
		staticCount: e.staticCount,
		shapeFlag: e.shapeFlag,
		patchFlag: t && e.type !== V ? o === -1 ? 16 : o | 16 : o,
		dynamicProps: e.dynamicProps,
		dynamicChildren: e.dynamicChildren,
		appContext: e.appContext,
		dirs: e.dirs,
		transition: c,
		component: e.component,
		suspense: e.suspense,
		ssContent: e.ssContent && ua(e.ssContent),
		ssFallback: e.ssFallback && ua(e.ssFallback),
		placeholder: e.placeholder,
		el: e.el,
		anchor: e.anchor,
		ctx: e.ctx,
		ce: e.ce
	};
	return c && r && ur(u, c.clone(u)), u;
}
function da(e = " ", t = 0) {
	return K(Yi, null, e, t);
}
function fa(e, t) {
	let n = K(Xi, null, e);
	return n.staticCount = t, n;
}
function q(e = "", t = !1) {
	return t ? (U(), ra(H, null, e)) : K(H, null, e);
}
function pa(e) {
	return e == null || typeof e == "boolean" ? K(H) : d(e) ? K(V, null, e.slice()) : ia(e) ? ma(e) : K(Yi, null, String(e));
}
function ma(e) {
	return e.el === null && e.patchFlag !== -1 || e.memo ? e : ua(e);
}
function ha(e, t) {
	let n = 0, { shapeFlag: r } = e;
	if (t == null) t = null;
	else if (d(t)) n = 16;
	else if (typeof t == "object") if (r & 65) {
		let n = t.default;
		n && (n._c && (n._d = !1), ha(e, n()), n._c && (n._d = !0));
		return;
	} else {
		n = 32;
		let r = t._;
		!r && !xi(t) ? t._ctx = wn : r === 3 && wn && (wn.slots._ === 1 ? t._ = 1 : (t._ = 2, e.patchFlag |= 1024));
	}
	else if (h(t)) {
		if (r & 65) {
			ha(e, { default: t });
			return;
		}
		t = {
			default: t,
			_ctx: wn
		}, n = 32;
	} else t = String(t), r & 64 ? (n = 16, t = [da(t)]) : n = 8;
	e.children = t, e.shapeFlag |= n;
}
function ga(...e) {
	let t = {};
	for (let n = 0; n < e.length; n++) {
		let r = e[n];
		for (let e in r) if (e === "class") t.class !== r.class && (t.class = O([t.class, r.class]));
		else if (e === "style") t.style = fe([t.style, r.style]);
		else if (a(e)) {
			let n = t[e], i = r[e];
			i && n !== i && !(d(n) && n.includes(i)) ? t[e] = n ? [].concat(n, i) : i : i == null && n == null && !o(e) && (t[e] = i);
		} else e !== "" && (t[e] = r[e]);
	}
	return t;
}
function _a(e, t, n, r = null) {
	on(e, t, 7, [n, r]);
}
var va = ri(), ya = 0;
function ba(e, n, r) {
	let i = e.type, a = (n ? n.appContext : e.appContext) || va, o = {
		uid: ya++,
		vnode: e,
		type: i,
		parent: n,
		appContext: a,
		root: null,
		next: null,
		subTree: null,
		effect: null,
		update: null,
		job: null,
		scope: new Ee(!0),
		render: null,
		proxy: null,
		exposed: null,
		exposeProxy: null,
		withProxy: null,
		provides: n ? n.provides : Object.create(a.provides),
		ids: n ? n.ids : [
			"",
			0,
			0
		],
		accessCache: null,
		renderCache: [],
		components: null,
		directives: null,
		propsOptions: Di(i, a),
		emitsOptions: ui(i, a),
		emit: null,
		emitted: null,
		propsDefaults: t,
		inheritAttrs: i.inheritAttrs,
		ctx: t,
		data: t,
		props: t,
		attrs: t,
		slots: t,
		refs: t,
		setupState: t,
		setupContext: null,
		suspense: r,
		suspenseId: r ? r.pendingId : 0,
		asyncDep: null,
		asyncResolved: !1,
		isMounted: !1,
		isUnmounted: !1,
		isDeactivated: !1,
		bc: null,
		c: null,
		bm: null,
		m: null,
		bu: null,
		u: null,
		um: null,
		bum: null,
		da: null,
		a: null,
		rtg: null,
		rtc: null,
		ec: null,
		sp: null
	};
	return o.ctx = { _: o }, o.root = n ? n.root : o, o.emit = ci.bind(null, o), e.ce && e.ce(o), o;
}
var J = null, xa = () => J || wn, Sa, Ca;
{
	let e = de(), t = (t, n) => {
		let r;
		return (r = e[t]) || (r = e[t] = []), r.push(n), (e) => {
			r.length > 1 ? r.forEach((t) => t(e)) : r[0](e);
		};
	};
	Sa = t("__VUE_INSTANCE_SETTERS__", (e) => J = e), Ca = t("__VUE_SSR_SETTERS__", (e) => Da = e);
}
var wa = (e) => {
	let t = J;
	return Sa(e), e.scope.on(), () => {
		e.scope.off(), Sa(t);
	};
}, Ta = () => {
	J && J.scope.off(), Sa(null);
};
function Ea(e) {
	return e.vnode.shapeFlag & 4;
}
var Da = !1;
function Oa(e, t = !1, n = !1) {
	t && Ca(t);
	let { props: r, children: i } = e.vnode, a = Ea(e);
	Si(e, r, a, t), Fi(e, i, n || t);
	let o = a ? ka(e, t) : void 0;
	return t && Ca(!1), o;
}
function ka(e, t) {
	let n = e.type;
	e.accessCache = /* @__PURE__ */ Object.create(null), e.proxy = new Proxy(e.ctx, Vr);
	let { setup: r } = n;
	if (r) {
		We();
		let n = e.setupContext = r.length > 1 ? Fa(e) : null, i = wa(e), a = an(r, e, 0, [e.props, n]), o = y(a);
		if (Ge(), i(), (o || e.sp) && !vr(e) && pr(e), o) {
			if (a.then(Ta, Ta), t) return a.then((n) => {
				Aa(e, n, t);
			}).catch((t) => {
				sn(t, e, 0);
			});
			e.asyncDep = a;
		} else Aa(e, a, t);
	} else Na(e, t);
}
function Aa(e, t, n) {
	h(t) ? e.type.__ssrInlineRender ? e.ssrRender = t : e.render = t : v(t) && (e.setupState = Yt(t)), Na(e, n);
}
var ja, Ma;
function Na(e, t, n) {
	let i = e.type;
	if (!e.render) {
		if (!t && ja && !i.render) {
			let t = i.template || Jr(e).template;
			if (t) {
				let { isCustomElement: n, compilerOptions: r } = e.appContext.config, { delimiters: a, compilerOptions: o } = i;
				i.render = ja(t, s(s({
					isCustomElement: n,
					delimiters: a
				}, r), o));
			}
		}
		e.render = i.render || r, Ma && Ma(e);
	}
	{
		let t = wa(e);
		We();
		try {
			Wr(e);
		} finally {
			Ge(), t();
		}
	}
}
var Pa = { get(e, t) {
	return M(e, "get", ""), e[t];
} };
function Fa(e) {
	return {
		attrs: new Proxy(e.attrs, Pa),
		slots: e.slots,
		emit: e.emit,
		expose: (t) => {
			e.exposed = t || {};
		}
	};
}
function Ia(e) {
	return e.exposed ? e.exposeProxy ||= new Proxy(Yt(Ut(e.exposed)), {
		get(t, n) {
			if (n in t) return t[n];
			if (n in zr) return zr[n](e);
		},
		has(e, t) {
			return t in e || t in zr;
		}
	}) : e.proxy;
}
function La(e) {
	return h(e) && "__vccOpts" in e;
}
var Y = (e, t) => /* @__PURE__ */ Zt(e, t, Da);
function Ra(e, t, n) {
	try {
		ta(-1);
		let r = arguments.length;
		return r === 2 ? v(t) && !d(t) ? ia(t) ? K(e, null, [t]) : K(e, t) : K(e, null, t) : (r > 3 ? n = Array.prototype.slice.call(arguments, 2) : r === 3 && ia(n) && (n = [n]), K(e, t, n));
	} finally {
		ta(1);
	}
}
var za = "3.5.40", Ba = void 0, Va = typeof window < "u" && window.trustedTypes;
if (Va) try {
	Ba = /* @__PURE__ */ Va.createPolicy("vue", { createHTML: (e) => e });
} catch {}
var Ha = Ba ? (e) => Ba.createHTML(e) : (e) => e, Ua = "http://www.w3.org/2000/svg", Wa = "http://www.w3.org/1998/Math/MathML", Ga = typeof document < "u" ? document : null, Ka = Ga && /* @__PURE__ */ Ga.createElement("template"), qa = {
	insert: (e, t, n) => {
		t.insertBefore(e, n || null);
	},
	remove: (e) => {
		let t = e.parentNode;
		t && t.removeChild(e);
	},
	createElement: (e, t, n, r) => {
		let i = t === "svg" ? Ga.createElementNS(Ua, e) : t === "mathml" ? Ga.createElementNS(Wa, e) : n ? Ga.createElement(e, { is: n }) : Ga.createElement(e);
		return e === "select" && r && r.multiple != null && i.setAttribute("multiple", r.multiple), i;
	},
	createText: (e) => Ga.createTextNode(e),
	createComment: (e) => Ga.createComment(e),
	setText: (e, t) => {
		e.nodeValue = t;
	},
	setElementText: (e, t) => {
		e.textContent = t;
	},
	parentNode: (e) => e.parentNode,
	nextSibling: (e) => e.nextSibling,
	querySelector: (e) => Ga.querySelector(e),
	setScopeId(e, t) {
		e.setAttribute(t, "");
	},
	insertStaticContent(e, t, n, r, i, a) {
		let o = n ? n.previousSibling : t.lastChild;
		if (i && (i === a || i.nextSibling)) for (; t.insertBefore(i.cloneNode(!0), n), !(i === a || !(i = i.nextSibling)););
		else {
			Ka.innerHTML = Ha(r === "svg" ? `<svg>${e}</svg>` : r === "mathml" ? `<math>${e}</math>` : e);
			let i = Ka.content;
			if (r === "svg" || r === "mathml") {
				let e = i.firstChild;
				for (; e.firstChild;) i.appendChild(e.firstChild);
				i.removeChild(e);
			}
			t.insertBefore(i, n);
		}
		return [o ? o.nextSibling : t.firstChild, n ? n.previousSibling : t.lastChild];
	}
}, Ja = "transition", Ya = "animation", Xa = /* @__PURE__ */ Symbol("_vtc"), Za = {
	name: String,
	type: String,
	css: {
		type: Boolean,
		default: !0
	},
	duration: [
		String,
		Number,
		Object
	],
	enterFromClass: String,
	enterActiveClass: String,
	enterToClass: String,
	appearFromClass: String,
	appearActiveClass: String,
	appearToClass: String,
	leaveFromClass: String,
	leaveActiveClass: String,
	leaveToClass: String
}, Qa = /* @__PURE__ */ s({}, tr, Za), $a = /* @__PURE__ */ ((e) => (e.displayName = "Transition", e.props = Qa, e))((e, { slots: t }) => Ra(ar, no(e), t)), eo = (e, t = []) => {
	d(e) ? e.forEach((e) => e(...t)) : e && e(...t);
}, to = (e) => e ? d(e) ? e.some((e) => e.length > 1) : e.length > 1 : !1;
function no(e) {
	let t = {};
	for (let n in e) n in Za || (t[n] = e[n]);
	if (e.css === !1) return t;
	let { name: n = "v", type: r, duration: i, enterFromClass: a = `${n}-enter-from`, enterActiveClass: o = `${n}-enter-active`, enterToClass: c = `${n}-enter-to`, appearFromClass: l = a, appearActiveClass: u = o, appearToClass: d = c, leaveFromClass: f = `${n}-leave-from`, leaveActiveClass: p = `${n}-leave-active`, leaveToClass: m = `${n}-leave-to` } = e, h = ro(i), g = h && h[0], _ = h && h[1], { onBeforeEnter: v, onEnter: y, onEnterCancelled: b, onLeave: x, onLeaveCancelled: S, onBeforeAppear: C = v, onAppear: w = y, onAppearCancelled: ee = b } = t, te = (e, t, n, r) => {
		e._enterCancelled = r, oo(e, t ? d : c), oo(e, t ? u : o), n && n();
	}, ne = (e, t) => {
		e._isLeaving = !1, oo(e, f), oo(e, m), oo(e, p), t && t();
	}, T = (e) => (t, n) => {
		let i = e ? w : y, o = () => te(t, e, n);
		eo(i, [t, o]), so(() => {
			oo(t, e ? l : a), ao(t, e ? d : c), to(i) || lo(t, r, g, o);
		});
	};
	return s(t, {
		onBeforeEnter(e) {
			eo(v, [e]), ao(e, a), ao(e, o);
		},
		onBeforeAppear(e) {
			eo(C, [e]), ao(e, l), ao(e, u);
		},
		onEnter: T(!1),
		onAppear: T(!0),
		onLeave(e, t) {
			e._isLeaving = !0;
			let n = () => ne(e, t);
			ao(e, f), e._enterCancelled ? (ao(e, p), mo(e)) : (mo(e), ao(e, p)), so(() => {
				e._isLeaving && (oo(e, f), ao(e, m), to(x) || lo(e, r, _, n));
			}), eo(x, [e, n]);
		},
		onEnterCancelled(e) {
			te(e, !1, void 0, !0), eo(b, [e]);
		},
		onAppearCancelled(e) {
			te(e, !0, void 0, !0), eo(ee, [e]);
		},
		onLeaveCancelled(e) {
			ne(e), eo(S, [e]);
		}
	});
}
function ro(e) {
	if (e == null) return null;
	if (v(e)) return [io(e.enter), io(e.leave)];
	{
		let t = io(e);
		return [t, t];
	}
}
function io(e) {
	return le(e);
}
function ao(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.add(t)), (e[Xa] || (e[Xa] = /* @__PURE__ */ new Set())).add(t);
}
function oo(e, t) {
	t.split(/\s+/).forEach((t) => t && e.classList.remove(t));
	let n = e[Xa];
	n && (n.delete(t), n.size || (e[Xa] = void 0));
}
function so(e) {
	requestAnimationFrame(() => {
		requestAnimationFrame(e);
	});
}
var co = 0;
function lo(e, t, n, r) {
	let i = e._endId = ++co, a = () => {
		i === e._endId && r();
	};
	if (n != null) return setTimeout(a, n);
	let { type: o, timeout: s, propCount: c } = uo(e, t);
	if (!o) return r();
	let l = o + "end", u = 0, d = () => {
		e.removeEventListener(l, f), a();
	}, f = (t) => {
		t.target === e && ++u >= c && d();
	};
	setTimeout(() => {
		u < c && d();
	}, s + 1), e.addEventListener(l, f);
}
function uo(e, t) {
	let n = window.getComputedStyle(e), r = (e) => (n[e] || "").split(", "), i = r(`${Ja}Delay`), a = r(`${Ja}Duration`), o = fo(i, a), s = r(`${Ya}Delay`), c = r(`${Ya}Duration`), l = fo(s, c), u = null, d = 0, f = 0;
	t === Ja ? o > 0 && (u = Ja, d = o, f = a.length) : t === Ya ? l > 0 && (u = Ya, d = l, f = c.length) : (d = Math.max(o, l), u = d > 0 ? o > l ? Ja : Ya : null, f = u ? u === Ja ? a.length : c.length : 0);
	let p = u === Ja && /\b(?:transform|all)(?:,|$)/.test(r(`${Ja}Property`).toString());
	return {
		type: u,
		timeout: d,
		propCount: f,
		hasTransform: p
	};
}
function fo(e, t) {
	for (; e.length < t.length;) e = e.concat(e);
	return Math.max(...t.map((t, n) => po(t) + po(e[n])));
}
function po(e) {
	return e === "auto" ? 0 : Number(e.slice(0, -1).replace(",", ".")) * 1e3;
}
function mo(e) {
	return (e ? e.ownerDocument : document).body.offsetHeight;
}
function ho(e, t, n) {
	let r = e[Xa];
	r && (t = (t ? [t, ...r] : [...r]).join(" ")), t == null ? e.removeAttribute("class") : n ? e.setAttribute("class", t) : e.className = t;
}
var go = /* @__PURE__ */ Symbol("_vod"), _o = /* @__PURE__ */ Symbol("_vsh"), vo = /* @__PURE__ */ Symbol(""), yo = /(?:^|;)\s*display\s*:/;
function bo(e, t, n) {
	let r = e.style, i = g(n), a = !1;
	if (n && !i) {
		if (t) if (g(t)) for (let e of t.split(";")) {
			let t = e.slice(0, e.indexOf(":")).trim();
			n[t] ?? So(r, t, "");
		}
		else for (let e in t) n[e] ?? So(r, e, "");
		for (let i in n) {
			i === "display" && (a = !0);
			let o = n[i];
			o == null ? So(r, i, "") : Eo(e, i, !g(t) && t ? t[i] : void 0, o) || So(r, i, o);
		}
	} else if (i) {
		if (t !== n) {
			let e = r[vo];
			e && (n += ";" + e), r.cssText = n, a = yo.test(n);
		}
	} else t && e.removeAttribute("style");
	go in e && (e[go] = a ? r.display : "", e[_o] && (r.display = "none"));
}
var xo = /\s*!important$/;
function So(e, t, n) {
	if (d(n)) n.forEach((n) => So(e, t, n));
	else if (n ??= "", t.startsWith("--")) e.setProperty(t, n);
	else {
		let r = To(e, t);
		xo.test(n) ? e.setProperty(E(r), n.replace(xo, ""), "important") : e[r] = n;
	}
}
var Co = [
	"Webkit",
	"Moz",
	"ms"
], wo = {};
function To(e, t) {
	let n = wo[t];
	if (n) return n;
	let r = T(t);
	if (r !== "filter" && r in e) return wo[t] = r;
	r = ie(r);
	for (let n = 0; n < Co.length; n++) {
		let i = Co[n] + r;
		if (i in e) return wo[t] = i;
	}
	return t;
}
function Eo(e, t, n, r) {
	return e.tagName === "TEXTAREA" && (t === "width" || t === "height") && g(r) && n === r;
}
var Do = "http://www.w3.org/1999/xlink";
function Oo(e, t, n, r, i, a = ve(t)) {
	r && t.startsWith("xlink:") ? n == null ? e.removeAttributeNS(Do, t.slice(6, t.length)) : e.setAttributeNS(Do, t, n) : n == null || a && !ye(n) ? e.removeAttribute(t) : e.setAttribute(t, a ? "" : _(n) ? String(n) : n);
}
function ko(e, t, n, r, i) {
	if (t === "innerHTML" || t === "textContent") {
		n != null && (e[t] = t === "innerHTML" ? Ha(n) : n);
		return;
	}
	let a = e.tagName;
	if (t === "value" && a !== "PROGRESS" && !a.includes("-")) {
		let r = a === "OPTION" ? e.getAttribute("value") || "" : e.value, i = n == null ? e.type === "checkbox" ? "on" : "" : String(n);
		(r !== i || !("_value" in e)) && (e.value = i), n ?? e.removeAttribute(t), e._value = n;
		return;
	}
	let o = !1;
	if (n === "" || n == null) {
		let r = typeof e[t];
		r === "boolean" ? n = ye(n) : n == null && r === "string" ? (n = "", o = !0) : r === "number" && (n = 0, o = !0);
	}
	try {
		e[t] = n;
	} catch {}
	o && e.removeAttribute(i || t);
}
function Ao(e, t, n, r) {
	e.addEventListener(t, n, r);
}
function jo(e, t, n, r) {
	e.removeEventListener(t, n, r);
}
var Mo = /* @__PURE__ */ Symbol("_vei");
function No(e, t, n, r, i = null) {
	let a = e[Mo] || (e[Mo] = {}), o = a[t];
	if (r && o) o.value = r;
	else {
		let [n, s] = Io(t);
		r ? Ao(e, n, a[t] = Bo(r, i), s) : o && (jo(e, n, o, s), a[t] = void 0);
	}
}
var Po = /(Once|Passive|Capture)$/, Fo = /^on:?(?:Once|Passive|Capture)$/;
function Io(e) {
	let t, n;
	for (; (n = e.match(Po)) && !Fo.test(e);) t ||= {}, e = e.slice(0, e.length - n[1].length), t[n[1].toLowerCase()] = !0;
	return [e[2] === ":" ? e.slice(3) : E(e.slice(2)), t];
}
var Lo = 0, Ro = /* @__PURE__ */ Promise.resolve(), zo = () => Lo ||= (Ro.then(() => Lo = 0), Date.now());
function Bo(e, t) {
	let n = (e) => {
		if (!e._vts) e._vts = Date.now();
		else if (e._vts <= n.attached) return;
		let r = n.value;
		if (d(r)) {
			let n = e.stopImmediatePropagation;
			e.stopImmediatePropagation = () => {
				n.call(e), e._stopped = !0;
			};
			let i = r.slice(), a = [e];
			for (let n = 0; n < i.length && !e._stopped; n++) {
				let e = i[n];
				e && on(e, t, 5, a);
			}
		} else on(r, t, 5, [e]);
	};
	return n.value = e, n.attached = zo(), n;
}
var Vo = (e) => e.charCodeAt(0) === 111 && e.charCodeAt(1) === 110 && e.charCodeAt(2) > 96 && e.charCodeAt(2) < 123, Ho = (e, t, n, r, i, s) => {
	let c = i === "svg";
	t === "class" ? ho(e, r, c) : t === "style" ? bo(e, n, r) : a(t) ? o(t) || No(e, t, n, r, s) : (t[0] === "." ? (t = t.slice(1), !0) : t[0] === "^" ? (t = t.slice(1), !1) : Uo(e, t, r, c)) ? (ko(e, t, r), !e.tagName.includes("-") && (t === "value" || t === "checked" || t === "selected") && Oo(e, t, r, c, s, t !== "value")) : e._isVueCE && (Wo(e, t) || e._def.__asyncLoader && (/[A-Z]/.test(t) || !g(r))) ? ko(e, T(t), r, s, t) : (t === "true-value" ? e._trueValue = r : t === "false-value" && (e._falseValue = r), Oo(e, t, r, c));
};
function Uo(e, t, n, r) {
	if (r) return !!(t === "innerHTML" || t === "textContent" || t in e && Vo(t) && h(n));
	if (t === "spellcheck" || t === "draggable" || t === "translate" || t === "autocorrect" || t === "sandbox" && e.tagName === "IFRAME" || t === "form" || t === "list" && e.tagName === "INPUT" || t === "type" && e.tagName === "TEXTAREA") return !1;
	if (t === "width" || t === "height") {
		let t = e.tagName;
		if (t === "IMG" || t === "VIDEO" || t === "CANVAS" || t === "SOURCE") return !1;
	}
	return Vo(t) && g(n) ? !1 : t in e;
}
function Wo(e, t) {
	let n = e._def.props;
	if (!n) return !1;
	let r = T(t);
	return Array.isArray(n) ? n.some((e) => T(e) === r) : Object.keys(n).some((e) => T(e) === r);
}
var Go = (e) => {
	let t = e.props["onUpdate:modelValue"] || !1;
	return d(t) ? (e) => se(t, e) : t;
};
function Ko(e) {
	e.target.composing = !0;
}
function qo(e) {
	let t = e.target;
	t.composing && (t.composing = !1, t.dispatchEvent(new Event("input")));
}
var Jo = /* @__PURE__ */ Symbol("_assign");
function Yo(e, t, n) {
	return t && (e = e.trim()), n && (e = ce(e)), e;
}
var Xo = {
	created(e, { modifiers: { lazy: t, trim: n, number: r } }, i) {
		e[Jo] = Go(i);
		let a = r || i.props && i.props.type === "number";
		Ao(e, t ? "change" : "input", (t) => {
			t.target.composing || e[Jo](Yo(e.value, n, a));
		}), (n || a) && Ao(e, "change", () => {
			e.value = Yo(e.value, n, a);
		}), t || (Ao(e, "compositionstart", Ko), Ao(e, "compositionend", qo), Ao(e, "change", qo));
	},
	mounted(e, { value: t }) {
		e.value = t ?? "";
	},
	beforeUpdate(e, { value: t, oldValue: n, modifiers: { lazy: r, trim: i, number: a } }, o) {
		if (e[Jo] = Go(o), e.composing) return;
		let s = (a || e.type === "number") && !/^0\d/.test(e.value) ? ce(e.value) : e.value, c = t ?? "";
		if (s === c) return;
		let l = e.getRootNode();
		(l instanceof Document || l instanceof ShadowRoot) && l.activeElement === e && e.type !== "range" && (r && t === n || i && e.value.trim() === c) || (e.value = c);
	}
}, Zo = {
	deep: !0,
	created(e, t, n) {
		e[Jo] = Go(n), Ao(e, "change", () => {
			let t = e._modelValue, n = ts(e), r = e.checked, i = e[Jo];
			if (d(t)) {
				let e = Se(t, n), a = e !== -1;
				if (r && !a) i(t.concat(n));
				else if (!r && a) {
					let n = [...t];
					n.splice(e, 1), i(n);
				}
			} else if (p(t)) {
				let e = new Set(t);
				r ? e.add(n) : e.delete(n), i(e);
			} else i(ns(e, r));
		});
	},
	mounted: Qo,
	beforeUpdate(e, t, n) {
		e[Jo] = Go(n), Qo(e, t, n);
	}
};
function Qo(e, { value: t, oldValue: n }, r) {
	e._modelValue = t;
	let i;
	if (d(t)) i = Se(t, r.props.value) > -1;
	else if (p(t)) i = t.has(r.props.value);
	else {
		if (t === n) return;
		i = xe(t, ns(e, !0));
	}
	e.checked !== i && (e.checked = i);
}
var $o = {
	deep: !0,
	created(e, { value: t, modifiers: { number: n } }, r) {
		e._modelValue = t, Ao(e, "change", () => {
			let t = Array.prototype.filter.call(e.options, (e) => e.selected).map((e) => n ? ce(ts(e)) : ts(e));
			e[Jo](e.multiple ? p(e._modelValue) ? new Set(t) : t : t[0]), e._assigning = !0, hn(() => {
				e._assigning = !1;
			});
		}), e[Jo] = Go(r);
	},
	mounted(e, { value: t }) {
		es(e, t);
	},
	beforeUpdate(e, { value: t }, n) {
		e._modelValue = t, e[Jo] = Go(n);
	},
	updated(e, { value: t }) {
		e._assigning || es(e, t);
	}
};
function es(e, t) {
	let n = e.multiple, r = d(t);
	if (!(n && !r && !p(t))) {
		for (let i = 0, a = e.options.length; i < a; i++) {
			let a = e.options[i], o = ts(a);
			if (n) if (r) {
				let e = typeof o;
				a.selected = e === "string" || e === "number" ? t.some((e) => String(e) === String(o)) : Se(t, o) > -1;
			} else a.selected = t.has(o);
			else if (xe(ts(a), t)) {
				e.selectedIndex !== i && (e.selectedIndex = i);
				return;
			}
		}
		!n && e.selectedIndex !== -1 && (e.selectedIndex = -1);
	}
}
function ts(e) {
	return "_value" in e ? e._value : e.value;
}
function ns(e, t) {
	let n = t ? "_trueValue" : "_falseValue";
	return n in e ? e[n] : t;
}
var rs = [
	"ctrl",
	"shift",
	"alt",
	"meta"
], is = {
	stop: (e) => e.stopPropagation(),
	prevent: (e) => e.preventDefault(),
	self: (e) => e.target !== e.currentTarget,
	ctrl: (e) => !e.ctrlKey,
	shift: (e) => !e.shiftKey,
	alt: (e) => !e.altKey,
	meta: (e) => !e.metaKey,
	left: (e) => "button" in e && e.button !== 0,
	middle: (e) => "button" in e && e.button !== 1,
	right: (e) => "button" in e && e.button !== 2,
	exact: (e, t) => rs.some((n) => e[`${n}Key`] && !t.includes(n))
}, as = (e, t) => {
	if (!e) return e;
	let n = e._withMods ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n, ...r) => {
		for (let e = 0; e < t.length; e++) {
			let r = is[t[e]];
			if (r && r(n, t)) return;
		}
		return e(n, ...r);
	}));
}, os = {
	esc: "escape",
	space: " ",
	up: "arrow-up",
	left: "arrow-left",
	right: "arrow-right",
	down: "arrow-down",
	delete: "backspace"
}, ss = (e, t) => {
	let n = e._withKeys ||= {}, r = t.join(".");
	return n[r] || (n[r] = ((n) => {
		if (!("key" in n)) return;
		let r = E(n.key);
		if (t.some((e) => e === r || os[e] === r)) return e(n);
	}));
}, cs = /* @__PURE__ */ s({ patchProp: Ho }, qa), ls;
function us() {
	return ls ||= Li(cs);
}
var ds = ((...e) => {
	let t = us().createApp(...e), { mount: n } = t;
	return t.mount = (e) => {
		let r = ps(e);
		if (!r) return;
		let i = t._component;
		!h(i) && !i.render && !i.template && (i.template = r.innerHTML), r.nodeType === 1 && (r.textContent = "");
		let a = n(r, !1, fs(r));
		return r instanceof Element && (r.removeAttribute("v-cloak"), r.setAttribute("data-v-app", "")), a;
	}, t;
});
function fs(e) {
	if (e instanceof SVGElement) return "svg";
	if (typeof MathMLElement == "function" && e instanceof MathMLElement) return "mathml";
}
function ps(e) {
	return g(e) ? document.querySelector(e) : e;
}
//#endregion
//#region src/api.ts
async function ms(e, t = {}) {
	let n = await fetch(e, {
		credentials: "same-origin",
		...t,
		headers: {
			Accept: "application/json",
			...t.headers || {}
		}
	}), r = (n.headers.get("content-type") || "").includes("application/json") ? await n.json() : await n.text();
	if (!n.ok) {
		let e = typeof r == "object" && r ? r.error || r.detail || r.message : r;
		throw Error(String(e || `Request failed (${n.status})`));
	}
	return r;
}
function hs(e) {
	return ms(e);
}
function gs(e, t = {}) {
	return ms(e, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
function _s(e, t) {
	return ms(e, {
		method: "PUT",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(t)
	});
}
//#endregion
//#region src/trainerStore.ts
var vs = () => ({
	running: !1,
	exit_code: null,
	log_lines: []
}), ys = () => ({
	personal: [],
	negative: [],
	personal_count: 0,
	negative_count: 0
}), bs = () => ({
	items: [],
	captured_count: 0,
	personal_count: 0,
	negative_count: 0
}), xs = () => ({
	items: [],
	total_size_bytes: 0,
	total_file_count: 0
}), Ss = () => ({
	enabled: !1,
	wake_phrase: "",
	language: "en",
	stt_engine: "faster_whisper",
	minimum_transcript_chars: 2,
	delete_confirmed_wakes: !1,
	promote_close_misses: !1,
	schedule_hours: 24,
	minimum_new_negatives: 3,
	advertised_base_url: "",
	tater_url: "http://127.0.0.1:8501",
	notify_satellites: !0
}), X = /* @__PURE__ */ Ft({
	activeView: "trainer",
	initialized: !1,
	busy: /* @__PURE__ */ new Set(),
	phrase: "",
	language: "en",
	ttsMode: "hybrid",
	languages: [{
		code: "en",
		label: "English (en)",
		engines: ["omnivoice"]
	}],
	session: {},
	samples: ys(),
	captured: bs(),
	training: vs(),
	auto: {},
	autoForm: Ss(),
	wakeWords: [],
	managedData: xs(),
	selectedFiles: [],
	sampleBucket: "personal",
	samplePage: {
		personal: 0,
		negative: 0
	},
	uploadProgress: 0,
	uploadLabel: "No upload in progress",
	uploadDetail: "Choose files and upload when you are ready.",
	consoleOpen: !1,
	taterLinkOpen: !1,
	trimItem: null,
	trimBucket: "personal",
	toast: {
		message: "",
		tone: "success",
		serial: 0
	}
}), Cs = 0, ws = 0, Ts = Y(() => Number(X.samples.personal_count ?? X.session.takes_received ?? 0)), Es = Y(() => Number(X.samples.negative_count ?? X.captured.negative_count ?? 0)), Ds = Y(() => X.languages.find((e) => e.code === X.language) || X.languages[0]), Os = Y(() => {
	let e = Ds.value?.engines?.length ? Ds.value.engines : ["omnivoice"], t = X.ttsMode === "piper" ? e.filter((e) => e === "piper") : X.ttsMode === "hybrid" ? e : e.filter((e) => e !== "piper"), n = {
		omnivoice: "OmniVoice",
		qwen3: "Qwen3",
		moss: "MOSS",
		piper: "Piper"
	}, r = X.ttsMode === "piper" ? "Legacy" : Ns(Ds.value?.quality || "experimental");
	return `${t.map((e) => n[e] || e).join(" + ") || "Unavailable"} · ${r}`;
}), ks = Y(() => !!(X.training.running || X.training.exit_code !== null || X.training.log_lines?.length)), As = Y(() => X.samples[X.sampleBucket] || []), js = Y(() => !!X.auto.trainer_link?.linked), Ms = Y(() => {
	let e = X.auto.stt_engines;
	return Array.isArray(e) && e.length ? e : [{
		id: "faster_whisper",
		label: "Faster Whisper"
	}, {
		id: "parakeet_onnx",
		label: "Parakeet ONNX"
	}];
});
function Ns(e) {
	return String(e || "").replaceAll("_", " ").replace(/\b\w/g, (e) => e.toUpperCase());
}
function Z(e) {
	return e ? X.busy.has(e) : X.busy.size > 0;
}
function Q(e, t) {
	t ? X.busy.add(e) : X.busy.delete(e);
}
function $(e, t = "success") {
	X.toast = {
		message: String(e || ""),
		tone: t,
		serial: X.toast.serial + 1
	};
}
function Ps(e, t) {
	$(e instanceof Error ? e.message : t, "error");
}
function Fs(e) {
	X.session = e || {}, Array.isArray(e.available_languages) && e.available_languages.length && (X.languages = e.available_languages), e.raw_phrase && (X.phrase = e.raw_phrase), e.language && (X.language = e.language), e.tts_mode && (X.ttsMode = e.tts_mode), e.training && (X.training = e.training);
}
async function Is() {
	let e = await hs("/api/session");
	return Fs(e), e;
}
async function Ls() {
	if (!X.phrase.trim()) {
		$("Enter a wake phrase first.", "warning");
		return;
	}
	Q("session", !0);
	try {
		let e = await gs("/api/start_session", {
			phrase: X.phrase.trim(),
			language: X.language,
			tts_mode: X.ttsMode
		});
		Fs(e), $(`Session ${e.safe_word || "started"} is ready.`);
	} catch (e) {
		Ps(e, "Session failed to start.");
	} finally {
		Q("session", !1);
	}
}
async function Rs() {
	let e = !!X.training.running;
	if (!(e && !window.confirm("Training is running. Stop training cleanly and end this session?"))) {
		Q("session", !0), ws &&= (window.clearInterval(ws), 0);
		try {
			Fs(await gs("/api/stop_session")), $(e ? "Training stopped cleanly and the session ended." : "Session ended. You can edit the wake phrase now.");
		} catch (t) {
			e && sc(), Ps(t, "Session could not be stopped.");
		} finally {
			Q("session", !1);
		}
	}
}
function zs() {
	if (!X.phrase.trim() || !("speechSynthesis" in window)) return;
	let e = new SpeechSynthesisUtterance(X.phrase.trim());
	e.lang = X.language, window.speechSynthesis.cancel(), window.speechSynthesis.speak(e);
}
function Bs() {
	let e = Ds.value?.engines || [], t = e.some((e) => e !== "piper"), n = e.includes("piper");
	X.ttsMode === "modern" && !t && (X.ttsMode = "piper"), X.ttsMode === "hybrid" && !(t && n) && (X.ttsMode = t ? "modern" : "piper"), X.ttsMode === "piper" && !n && (X.ttsMode = "modern");
}
async function Vs(e = !1) {
	e || Q("samples", !0);
	try {
		let e = await hs("/api/samples");
		X.samples = {
			...ys(),
			...e
		};
		for (let e of ["personal", "negative"]) {
			let t = Math.max(0, Math.ceil((X.samples[e]?.length || 0) / 50) - 1);
			X.samplePage[e] = Math.min(X.samplePage[e], t);
		}
		return e;
	} finally {
		e || Q("samples", !1);
	}
}
async function Hs(e = !1) {
	e || Q("captured", !0);
	try {
		let e = await hs("/api/captured_audio");
		return X.captured = {
			...bs(),
			...e
		}, e;
	} finally {
		e || Q("captured", !1);
	}
}
function Us(e) {
	let t = e.target;
	X.selectedFiles = Array.from(t.files || []);
}
function Ws(e, t, n) {
	return new Promise((r, i) => {
		let a = new XMLHttpRequest(), o = new FormData();
		o.append("file", e, e.name), a.open("POST", "/api/upload_personal_sample"), a.responseType = "json", a.upload.onprogress = (r) => {
			r.lengthComputable && (X.uploadProgress = Math.round((t + r.loaded / r.total) / n * 100), X.uploadLabel = `Uploading ${e.name} (${t + 1}/${n})`, X.uploadDetail = "Sending and normalizing the recording.");
		}, a.onload = () => {
			let t = a.response || {};
			a.status >= 200 && a.status < 300 ? r(t) : i(Error(t.error || `Upload failed for ${e.name}`));
		}, a.onerror = () => i(/* @__PURE__ */ Error(`Upload failed for ${e.name}`)), a.send(o);
	});
}
async function Gs(e) {
	if (!X.session.safe_word) {
		$("Start a trainer session before uploading samples.", "warning");
		return;
	}
	if (X.selectedFiles.length) {
		Q("upload", !0);
		try {
			let t = [...X.selectedFiles];
			for (let e = 0; e < t.length; e += 1) await Ws(t[e], e, t.length);
			X.uploadProgress = 100, X.uploadLabel = "Upload complete", X.uploadDetail = `${t.length} sample${t.length === 1 ? "" : "s"} saved in the required training format.`, X.selectedFiles = [], e && (e.value = ""), await Promise.all([Is(), Vs(!0)]), $("Personal samples uploaded.");
		} catch (e) {
			X.uploadProgress = 0, Ps(e, "Sample upload failed.");
		} finally {
			Q("upload", !1);
		}
	}
}
async function Ks(e, t) {
	if (!(t === "discard" && !window.confirm(`Discard ${e.saved_as} from the captured-audio inbox?`))) {
		Q("review", !0);
		try {
			await gs(`/api/captured_audio/${encodeURIComponent(e.saved_as)}/${t}`), await Promise.all([
				Is(),
				Hs(!0),
				Vs(!0)
			]), $(t === "approve_personal" ? "Clip added to personal samples." : t === "mark_negative" ? "Clip marked negative." : "Clip discarded.");
		} catch (e) {
			Ps(e, "Review action failed.");
		} finally {
			Q("review", !1);
		}
	}
}
async function qs(e, t) {
	if (window.confirm(`Remove ${e.saved_as} from ${t} samples?`)) {
		Q("review", !0);
		try {
			await ms(`/api/samples/${t}/${encodeURIComponent(e.saved_as)}`, { method: "DELETE" }), await Vs(!0), $("Sample removed.");
		} catch (e) {
			Ps(e, "Sample removal failed.");
		} finally {
			Q("review", !1);
		}
	}
}
async function Js(e, t) {
	if (!window.confirm(`Revert ${e.saved_as} to its pre-trim version?`)) return;
	let n = new FormData();
	n.append("bucket", t), n.append("file_name", e.saved_as), Q("review", !0);
	try {
		await ms("/api/samples/revert", {
			method: "POST",
			body: n
		}), await Vs(!0), $("Original sample restored.");
	} catch (e) {
		Ps(e, "Sample revert failed.");
	} finally {
		Q("review", !1);
	}
}
async function Ys(e) {
	let t = e === "personal" ? Ts.value : Es.value;
	if (!(!t || !window.confirm(`Clear ${t} ${e} sample${t === 1 ? "" : "s"}?`))) {
		Q("review", !0);
		try {
			await gs(e === "personal" ? "/api/reset_recordings" : "/api/reset_negative_samples"), await Promise.all([
				Is(),
				Vs(!0),
				Hs(!0)
			]), $(`${Ns(e)} samples cleared.`);
		} catch (e) {
			Ps(e, "Samples could not be cleared.");
		} finally {
			Q("review", !1);
		}
	}
}
function Xs(e, t) {
	X.auto = e || {}, t && (X.autoForm = {
		...Ss(),
		...e.config || {}
	}, X.autoForm.wake_phrase || (X.autoForm.wake_phrase = X.session.raw_phrase || ""), X.autoForm.language || (X.autoForm.language = X.session.language || "en"));
}
async function Zs(e = !1) {
	let t = await hs("/api/auto_train");
	return Xs(t, e), t;
}
async function Qs() {
	Q("auto", !0);
	try {
		let e = await _s("/api/auto_train", X.autoForm);
		Xs(e, !0), $(e.config?.enabled ? "Auto Training saved and enabled." : "Auto Training saved.");
	} catch (e) {
		Ps(e, "Auto Training settings failed to save.");
	} finally {
		Q("auto", !1);
	}
}
async function $s(e) {
	Q("auto", !0);
	try {
		let t = await gs("/api/auto_train/action", { action: e });
		Xs(t, !1), e === "train_now" && (X.consoleOpen = !0, sc()), $(e === "review_now" ? `${Number(t.queued || 0)} clips queued for review.` : e === "train_now" ? "Training started." : "Wake word published.");
	} catch (e) {
		Ps(e, "Auto Training action failed.");
	} finally {
		Q("auto", !1);
	}
}
async function ec(e, t) {
	Q("link", !0);
	try {
		return await gs("/api/tater_link/claim", {
			tater_url: e.trim(),
			pairing_code: t.trim()
		}), X.autoForm.tater_url = e.trim(), await Zs(!1), $("Trainer linked securely to Tater."), !0;
	} catch (e) {
		return Ps(e, "Tater link failed."), !1;
	} finally {
		Q("link", !1);
	}
}
async function tc() {
	if (window.confirm("Unlink this trainer from Tater?")) {
		Q("auto", !0);
		try {
			await gs("/api/tater_link/unlink"), await Zs(!1), $("Trainer unlinked from Tater.", "warning");
		} catch (e) {
			Ps(e, "Tater unlink failed.");
		} finally {
			Q("auto", !1);
		}
	}
}
async function nc(e = !1) {
	e || Q("firmware", !0);
	try {
		let e = await hs("/api/trained_wake_words/catalog");
		X.wakeWords = Array.isArray(e.wake_words) ? e.wake_words : [];
	} finally {
		e || Q("firmware", !1);
	}
}
async function rc() {
	Q("data", !0);
	try {
		let e = await hs("/api/data");
		return X.managedData = {
			...xs(),
			...e
		}, e;
	} finally {
		Q("data", !1);
	}
}
async function ic(e) {
	if (!e.file_count) return;
	let t = `${dc(e.size_bytes)} · ${Number(e.file_count).toLocaleString()} file${e.file_count === 1 ? "" : "s"}`, n = e.rebuild_note ? `\n\n${e.rebuild_note}` : "";
	if (window.confirm(`Permanently delete ${e.label} (${t})?${n}\n\nThis cannot be undone.`)) {
		Q("data-delete", !0);
		try {
			let t = await ms(`/api/data/${encodeURIComponent(e.id)}`, { method: "DELETE" });
			X.managedData = {
				...xs(),
				...t
			}, await Promise.allSettled([
				Is(),
				Vs(!0),
				Hs(!0),
				nc(!0)
			]), $(`${e.label} deleted. ${dc(e.size_bytes)} released.`);
		} catch (t) {
			Ps(t, `${e.label} could not be deleted.`);
		} finally {
			Q("data-delete", !1);
		}
	}
}
async function ac(e) {
	try {
		await navigator.clipboard.writeText(e), $("Wake-word JSON URL copied.");
	} catch (e) {
		Ps(e, "Clipboard unavailable.");
	}
}
async function oc() {
	await Promise.all([Is(), Vs(!0)]);
	let e = !1;
	if (!(!Ts.value && (e = window.confirm("No positive samples are saved. Train anyway without personal voices?"), !e))) {
		Q("training-start", !0), X.training = {
			running: !0,
			exit_code: null,
			log_lines: ["Waiting for training output…"]
		}, X.consoleOpen = !0;
		try {
			await gs("/api/train", { allow_no_personal: e }), sc();
		} catch (e) {
			X.training = {
				running: !1,
				exit_code: 1,
				log_lines: [e instanceof Error ? e.message : String(e)]
			}, Ps(e, "Training could not start.");
		} finally {
			Q("training-start", !1);
		}
	}
}
function sc() {
	if (ws) return;
	let e = async () => {
		try {
			let e = await hs("/api/train_status");
			X.training = {
				...vs(),
				...e.training || {}
			}, X.training.running || (window.clearInterval(ws), ws = 0, await Promise.all([Vs(!0), nc(!0)]), $(X.training.exit_code === 0 ? "Training finished successfully." : `Training ended with exit ${X.training.exit_code}.`, X.training.exit_code === 0 ? "success" : "error"));
		} catch {}
	};
	e(), ws = window.setInterval(() => void e(), 1500);
}
async function cc() {
	Q("bootstrap", !0);
	try {
		await Promise.allSettled([
			Is(),
			Vs(!0),
			Hs(!0),
			Zs(!0),
			nc(!0)
		]), Bs();
		try {
			let e = await hs("/api/train_status");
			X.training = {
				...vs(),
				...e.training || {}
			}, X.training.running && (X.consoleOpen = !0, sc());
		} catch {}
		Cs = window.setInterval(() => {
			X.activeView === "auto" && !Z("auto") && Zs(!1).catch(() => void 0);
		}, 2500), X.initialized = !0;
	} finally {
		Q("bootstrap", !1);
	}
}
function lc() {
	window.clearInterval(Cs), window.clearInterval(ws), Cs = 0, ws = 0;
}
function uc(e) {
	if (!e) return "";
	let t = new Date(String(e));
	return Number.isNaN(t.getTime()) ? String(e) : t.toLocaleString();
}
function dc(e) {
	let t = Math.max(0, Number(e) || 0);
	if (t < 1024) return `${Math.round(t)} B`;
	let n = [
		"KB",
		"MB",
		"GB",
		"TB"
	], r = t / 1024, i = n[0];
	for (let e = 1; e < n.length && r >= 1024; e += 1) r /= 1024, i = n[e];
	return `${r >= 10 ? r.toFixed(1) : r.toFixed(2)} ${i}`;
}
function fc(e) {
	if (!e) return "16 kHz · mono · 16-bit WAV";
	let t = Number(e.sample_rate || e.sample_rate_hz || 16e3), n = Number(e.channels || 1) === 1 ? "mono" : `${e.channels} channels`, r = Number(e.bits_per_sample || e.sample_width_bits || 16);
	return `${Math.round(t / 1e3)} kHz · ${n} · ${r}-bit`;
}
function pc(e) {
	if (e.blocked_by_vad) return {
		label: "Blocked by VAD",
		tone: "warning"
	};
	let t = String(e.event_type || "").toLowerCase();
	return t.includes("close") ? {
		label: e.capture_label || "Close miss",
		tone: "warning"
	} : t.includes("false") ? {
		label: e.capture_label || "False trigger",
		tone: "error"
	} : t.includes("wake") || t.includes("detect") ? {
		label: e.capture_label || "Wake trigger",
		tone: "success"
	} : {
		label: e.capture_label || "Captured",
		tone: "neutral"
	};
}
function mc(e, t) {
	return e.audio_url || `/api/audio/${t}/${encodeURIComponent(e.saved_as)}`;
}
//#endregion
//#region src/components/AudioTrimModal.vue?vue&type=script&setup=true&lang.ts
var hc = {
	class: "modal trim-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Trim audio"
}, gc = { class: "modal-head" }, _c = {
	key: 0,
	class: "empty-state"
}, vc = { class: "range-grid" }, yc = ["max"], bc = ["min", "max"], xc = { class: "row space" }, Sc = { class: "pill" }, Cc = {
	key: 0,
	class: "pill success"
}, wc = { class: "row modal-actions" }, Tc = ["disabled"], Ec = /* @__PURE__ */ fr({
	__name: "AudioTrimModal",
	setup(e) {
		let t = /* @__PURE__ */ F(null), n = /* @__PURE__ */ F(null), r = /* @__PURE__ */ F(0), i = /* @__PURE__ */ F(0), a = /* @__PURE__ */ F(0), o = /* @__PURE__ */ F([]), s = /* @__PURE__ */ F(!1), c = /* @__PURE__ */ F(!1);
		Nn(() => X.trimItem, async (e) => {
			if (!e) {
				n.value = null;
				return;
			}
			s.value = !0;
			try {
				let t = `/api/audio/${encodeURIComponent(X.trimBucket)}/${encodeURIComponent(e.saved_as)}`, s = await fetch(t);
				if (!s.ok) throw Error("Audio could not be loaded.");
				let c = new (window.AudioContext || window.webkitAudioContext)();
				n.value = await c.decodeAudioData(await s.arrayBuffer()), r.value = n.value.duration, i.value = 0, a.value = r.value, await c.close();
				try {
					let t = await ms(`/api/samples/${encodeURIComponent(X.trimBucket)}/${encodeURIComponent(e.saved_as)}/vad`, { method: "POST" });
					o.value = Array.isArray(t.segments) ? t.segments : [], o.value.length && (i.value = Math.max(0, Number(o.value[0].start || 0)), a.value = Math.min(r.value, Number(o.value[0].end || r.value)));
				} catch {
					o.value = [];
				}
				await hn(), d();
			} catch (e) {
				$(e instanceof Error ? e.message : "Audio could not be loaded.", "error"), l();
			} finally {
				s.value = !1;
			}
		}, { immediate: !0 }), Nn([i, a], () => d());
		function l() {
			X.trimItem = null, n.value = null, o.value = [];
		}
		function u() {
			let e = o.value[0];
			e && (i.value = Number(e.start), a.value = Number(e.end));
		}
		function d() {
			let e = t.value, s = n.value;
			if (!e || !s || !r.value) return;
			let c = e.getBoundingClientRect();
			if (!c.width || !c.height) return;
			let l = window.devicePixelRatio || 1;
			e.width = Math.round(c.width * l), e.height = Math.round(c.height * l);
			let u = e.getContext("2d");
			if (!u) return;
			u.scale(l, l);
			let d = c.width, f = c.height, p = f / 2, m = s.getChannelData(0), h = Math.max(1, Math.floor(m.length / d));
			u.clearRect(0, 0, d, f), u.strokeStyle = "rgba(222, 218, 212, .24)", u.lineWidth = 1, u.beginPath();
			for (let e = 0; e < d; e += 1) {
				let t = 1, n = -1;
				for (let r = 0; r < h; r += 1) {
					let i = m[Math.floor(e) * h + r] || 0;
					t = Math.min(t, i), n = Math.max(n, i);
				}
				u.moveTo(e, p + t * p * .84), u.lineTo(e, p + n * p * .84);
			}
			u.stroke();
			let g = i.value / r.value * d, _ = a.value / r.value * d;
			u.fillStyle = "rgba(8, 8, 9, .66)", u.fillRect(0, 0, g, f), u.fillRect(_, 0, d - _, f), u.fillStyle = "rgba(255, 145, 52, .12)", u.fillRect(g, 0, _ - g, f), u.strokeStyle = "#ff9134", u.lineWidth = 2;
			for (let e of [g, _]) u.beginPath(), u.moveTo(e, 0), u.lineTo(e, f), u.stroke();
			u.strokeStyle = "rgba(68, 225, 165, .55)";
			for (let e of o.value) {
				let t = e.start / r.value * d;
				u.beginPath(), u.moveTo(t, 0), u.lineTo(t, f), u.stroke();
			}
		}
		function f() {
			let e = n.value;
			if (!e) return;
			let t = new (window.AudioContext || window.webkitAudioContext)(), r = t.createBufferSource();
			r.buffer = e, r.connect(t.destination), r.start(0, i.value, Math.max(.01, a.value - i.value)), r.onended = () => void t.close();
		}
		async function p() {
			let e = n.value;
			if (!e) throw Error("Audio is not loaded.");
			let t = Math.floor(i.value * e.sampleRate), r = Math.min(Math.floor(a.value * e.sampleRate), e.length), o = 16e3, s;
			if (e.sampleRate === o) s = e.getChannelData(0).slice(t, r);
			else {
				let n = Math.max(1, Math.floor((r - t) * o / e.sampleRate)), c = new OfflineAudioContext(1, n, o), l = c.createBufferSource();
				l.buffer = e, l.connect(c.destination), l.start(0, i.value, a.value - i.value), s = (await c.startRendering()).getChannelData(0);
			}
			let c = /* @__PURE__ */ new ArrayBuffer(44 + s.length * 2), l = new DataView(c);
			l.setUint32(0, 1380533830, !1), l.setUint32(4, 36 + s.length * 2, !0), l.setUint32(8, 1463899717, !1), l.setUint32(12, 1718449184, !1), l.setUint32(16, 16, !0), l.setUint16(20, 1, !0), l.setUint16(22, 1, !0), l.setUint32(24, o, !0), l.setUint32(28, o * 2, !0), l.setUint16(32, 2, !0), l.setUint16(34, 16, !0), l.setUint32(36, 1684108385, !1), l.setUint32(40, s.length * 2, !0);
			for (let e = 0; e < s.length; e += 1) l.setInt16(44 + e * 2, Math.max(-32768, Math.min(32767, Math.round(s[e] * 32767))), !0);
			return new Blob([c], { type: "audio/wav" });
		}
		async function m() {
			let e = X.trimItem;
			if (e) {
				c.value = !0;
				try {
					let t = new FormData();
					t.append("file", await p(), "trimmed.wav"), t.append("bucket", X.trimBucket), t.append("source_file", e.saved_as), t.append("start_time", i.value.toFixed(3)), t.append("end_time", a.value.toFixed(3));
					let n = await ms("/api/samples/trim", {
						method: "POST",
						body: t
					});
					l(), await Vs(!0), $(n.message || "Trimmed sample saved.");
				} catch (e) {
					$(e instanceof Error ? e.message : "Trim failed.", "error");
				} finally {
					c.value = !1;
				}
			}
		}
		function h() {
			X.trimItem && d();
		}
		return window.addEventListener("resize", h), Ar(() => window.removeEventListener("resize", h)), (e, n) => (U(), ra(Jn, { to: "body" }, [I(X).trimItem ? (U(), W("div", {
			key: 0,
			class: "modal-backdrop",
			onClick: as(l, ["self"])
		}, [G("section", hc, [
			G("header", gc, [G("div", null, [n[2] ||= G("span", { class: "eyebrow" }, "Audio editor", -1), G("h2", null, "Trim " + k(I(X).trimItem.saved_as), 1)]), G("button", {
				type: "button",
				class: "button ghost",
				onClick: l
			}, "Close")]),
			n[3] ||= G("p", { class: "muted" }, "Keep the spoken wake phrase and remove excess silence or noise. VAD markers appear in green.", -1),
			s.value ? (U(), W("div", _c, "Loading waveform…")) : (U(), W(V, { key: 1 }, [
				G("canvas", {
					ref_key: "canvas",
					ref: t,
					class: "waveform"
				}, null, 512),
				G("div", vc, [G("label", null, [G("span", null, "Start · " + k(i.value.toFixed(2)) + "s", 1), R(G("input", {
					"onUpdate:modelValue": n[0] ||= (e) => i.value = e,
					type: "range",
					min: "0",
					max: Math.max(0, a.value - .01),
					step: ".01"
				}, null, 8, yc), [[
					Xo,
					i.value,
					void 0,
					{ number: !0 }
				]])]), G("label", null, [G("span", null, "End · " + k(a.value.toFixed(2)) + "s", 1), R(G("input", {
					"onUpdate:modelValue": n[1] ||= (e) => a.value = e,
					type: "range",
					min: Math.min(r.value, i.value + .01),
					max: r.value,
					step: ".01"
				}, null, 8, bc), [[
					Xo,
					a.value,
					void 0,
					{ number: !0 }
				]])])]),
				G("div", xc, [G("span", Sc, "Selection " + k(Math.max(0, a.value - i.value).toFixed(2)) + "s", 1), o.value.length ? (U(), W("span", Cc, k(o.value.length) + " speech segment" + k(o.value.length === 1 ? "" : "s"), 1)) : q("", !0)]),
				G("div", wc, [
					G("button", {
						type: "button",
						onClick: f
					}, "Play selection"),
					o.value.length ? (U(), W("button", {
						key: 0,
						type: "button",
						onClick: u
					}, "Select first VAD")) : q("", !0),
					G("button", {
						type: "button",
						class: "button primary",
						disabled: c.value,
						onClick: m
					}, k(c.value ? "Saving…" : "Save trim"), 9, Tc)
				])
			], 64))
		])])) : q("", !0)]));
	}
}), Dc = { class: "app-shell" }, Oc = { class: "app-header" }, kc = { class: "header-status" }, Ac = {
	key: 0,
	class: "session-chip"
}, jc = {
	class: "tabs",
	"aria-label": "Trainer areas"
}, Mc = ["onClick"], Nc = { class: "tab-full" }, Pc = { class: "tab-short" }, Fc = { key: 0 }, Ic = { class: "main-content" }, Lc = {
	key: 0,
	class: "loading-panel"
}, Rc = { class: "panel" }, zc = { class: "panel-head" }, Bc = { class: "form-grid phrase-form" }, Vc = { class: "field wide" }, Hc = ["disabled"], Uc = { class: "field" }, Wc = ["disabled"], Gc = ["value"], Kc = { class: "field" }, qc = ["disabled"], Jc = ["disabled"], Yc = ["disabled"], Xc = ["disabled"], Zc = { class: "row form-actions" }, Qc = ["disabled"], $c = ["disabled"], el = ["disabled"], tl = { class: "panel" }, nl = { class: "panel-head" }, rl = { class: "stats" }, il = { class: "train-action" }, al = ["disabled"], ol = { class: "panel-footer" }, sl = ["disabled"], cl = { class: "hero auto-hero" }, ll = { class: "panel" }, ul = { class: "toggle-list" }, dl = { class: "form-grid" }, fl = { class: "field" }, pl = { class: "field" }, ml = { class: "field wide" }, hl = ["value"], gl = { class: "field" }, _l = { class: "panel" }, vl = { class: "form-grid" }, yl = { class: "field" }, bl = { class: "field" }, xl = { class: "stats" }, Sl = { class: "format-value" }, Cl = { class: "format-value" }, wl = { class: "panel" }, Tl = { class: "form-grid" }, El = { class: "field wide" }, Dl = { class: "field wide" }, Ol = { class: "link-row" }, kl = ["disabled"], Al = ["disabled"], jl = { class: "toggle-list compact" }, Ml = { class: "panel action-panel" }, Nl = { class: "action-grid" }, Pl = ["disabled"], Fl = ["disabled"], Il = ["disabled"], Ll = ["disabled"], Rl = { class: "audit" }, zl = { class: "hero capture-hero" }, Bl = { class: "panel" }, Vl = { class: "panel-head" }, Hl = ["disabled"], Ul = { class: "stats" }, Wl = { class: "panel" }, Gl = {
	key: 0,
	class: "empty-state"
}, Kl = {
	key: 1,
	class: "audio-list"
}, ql = {
	key: 0,
	class: "meta-row"
}, Jl = {
	key: 1,
	class: "transcript"
}, Yl = {
	key: 2,
	class: "transcript"
}, Xl = ["src"], Zl = ["disabled", "onClick"], Ql = ["disabled", "onClick"], $l = ["disabled", "onClick"], eu = { class: "hero samples-hero" }, tu = { class: "pill hero-pill" }, nu = { class: "panel" }, ru = { class: "panel-head sample-head" }, iu = { class: "segment-control" }, au = { class: "row toolbar" }, ou = ["disabled"], su = ["disabled"], cu = ["disabled"], lu = {
	key: 0,
	class: "empty-state"
}, uu = {
	key: 1,
	class: "audio-list compact-list"
}, du = { class: "row" }, fu = {
	key: 0,
	class: "pill warning"
}, pu = {
	key: 0,
	class: "transcript"
}, mu = {
	key: 1,
	class: "transcript"
}, hu = ["src"], gu = ["onClick"], _u = ["onClick"], vu = ["disabled", "onClick"], yu = {
	key: 2,
	class: "pagination"
}, bu = ["disabled"], xu = ["disabled"], Su = { class: "panel" }, Cu = { class: "dropzone" }, wu = ["disabled"], Tu = { class: "progress-card" }, Eu = { class: "progress-track" }, Du = { class: "hero data-hero" }, Ou = { class: "pill hero-pill" }, ku = { class: "panel" }, Au = { class: "panel-head" }, ju = ["disabled"], Mu = { class: "stats" }, Nu = { class: "format-value" }, Pu = {
	key: 0,
	class: "data-warning"
}, Fu = { class: "panel-head" }, Iu = { class: "number" }, Lu = { class: "data-list" }, Ru = { class: "data-copy" }, zu = { class: "data-title" }, Bu = {
	key: 0,
	class: "data-note"
}, Vu = { class: "data-usage" }, Hu = ["disabled", "onClick"], Uu = {
	key: 0,
	class: "panel empty-state"
}, Wu = { class: "hero firmware-hero" }, Gu = { class: "panel" }, Ku = { class: "panel-head" }, qu = ["disabled"], Ju = {
	key: 0,
	class: "empty-state"
}, Yu = {
	key: 1,
	class: "word-list"
}, Xu = ["href"], Zu = {
	key: 1,
	class: "muted"
}, Qu = ["href"], $u = { class: "meta-row" }, ed = { key: 0 }, td = { key: 1 }, nd = { key: 2 }, rd = ["disabled", "onClick"], id = { class: "panel compatibility-panel" }, ad = {
	key: 0,
	class: "empty-state"
}, od = {
	key: 1,
	class: "word-list"
}, sd = ["href"], cd = {
	key: 1,
	class: "muted"
}, ld = ["disabled", "onClick"], ud = {
	class: "modal console-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Training console"
}, dd = { class: "modal-head" }, fd = { class: "row console-actions" }, pd = {
	class: "modal link-modal",
	role: "dialog",
	"aria-modal": "true",
	"aria-label": "Link Tater"
}, md = { class: "modal-head" }, hd = {
	key: 0,
	class: "link-success"
}, gd = {
	key: 1,
	class: "stack"
}, _d = { class: "field" }, vd = { class: "field" }, yd = ["disabled"], bd = "/static/images/tater-wake-word-trainer.png", xd = 50, Sd = /* @__PURE__ */ fr({
	__name: "TrainerApp",
	setup(e) {
		let t = /* @__PURE__ */ F(null), n = /* @__PURE__ */ F(null), r = /* @__PURE__ */ F(!0), i = /* @__PURE__ */ F(""), a = /* @__PURE__ */ F(""), o = /* @__PURE__ */ F(!1), s = [
			{
				id: "trainer",
				label: "Trainer",
				short: "Train"
			},
			{
				id: "auto",
				label: "Auto Training",
				short: "Auto"
			},
			{
				id: "firmware",
				label: "Wake Words",
				short: "Words"
			},
			{
				id: "captured",
				label: "Captured Audio",
				short: "Inbox"
			},
			{
				id: "samples",
				label: "Samples",
				short: "Samples"
			},
			{
				id: "data",
				label: "Data",
				short: "Data"
			}
		], c = Y(() => {
			let e = X.samplePage[X.sampleBucket];
			return As.value.slice(e * xd, (e + 1) * xd);
		}), l = Y(() => Math.max(1, Math.ceil(As.value.length / xd))), u = Y(() => X.auto.state || {}), d = Y(() => X.auto.runtime || {}), f = Y(() => {
			let e = u.value, t = [];
			return e.last_review_result && t.push(`Last review: ${String(e.last_review_result).replaceAll("_", " ")}`), e.last_review_file && t.push(String(e.last_review_file)), e.last_review_transcript && t.push(`STT: “${e.last_review_transcript}”`), e.last_review_error && t.push(`Error: ${e.last_review_error}`), e.last_stt_engine && t.push(`STT engine: ${String(e.last_stt_engine).replaceAll("_", " ")}`), e.last_notify_at && t.push(e.last_notify_error ? `Publish failed: ${e.last_notify_error}` : `Wake word published ${uc(e.last_notify_at)}`), t.join(" · ") || "No automatic review has run yet.";
		}), p = Y(() => X.training.running ? {
			text: "Training running",
			tone: "warning"
		} : X.training.exit_code === 0 ? {
			text: "Training finished",
			tone: "success"
		} : X.training.exit_code === null ? {
			text: "Not started",
			tone: "neutral"
		} : {
			text: `Exit ${X.training.exit_code}`,
			tone: "error"
		}), m = Y(() => d.value.review_running ? {
			text: `Transcribing ${d.value.review_file || "wake"}`,
			tone: "warning"
		} : X.training.running && X.auto.config?.enabled ? {
			text: "Training running",
			tone: "warning"
		} : X.auto.config?.enabled ? {
			text: "Enabled",
			tone: "success"
		} : {
			text: "Disabled",
			tone: "neutral"
		}), h = Y(() => X.training.log_lines?.length ? X.training.log_lines : ["No training output yet."]), g = Y(() => {
			let e = /* @__PURE__ */ new Map();
			for (let t of X.managedData.items || []) {
				let n = e.get(t.category) || [];
				n.push(t), e.set(t.category, n);
			}
			return Array.from(e, ([e, t]) => ({
				name: e,
				items: t
			}));
		});
		Nn(() => X.language, Bs), Nn(() => X.toast.serial, () => window.setTimeout(() => {
			X.toast.message = "";
		}, 4500)), Nn(h, async () => {
			r.value && (await hn(), r.value && n.value && (n.value.scrollTop = n.value.scrollHeight));
		}), Nn(() => X.consoleOpen, async (e) => {
			e && (r.value = !0, await hn(), y());
		}), Dr(() => {
			cc(), document.addEventListener("keydown", _);
		}), Ar(() => {
			lc(), document.removeEventListener("keydown", _);
		});
		function _(e) {
			e.key === "Escape" && (X.consoleOpen = !1, X.taterLinkOpen = !1, X.trimItem = null);
		}
		function v() {
			let e = n.value;
			if (!e) return;
			let t = e.scrollHeight - e.clientHeight - e.scrollTop;
			r.value = t <= 32;
		}
		function y() {
			let e = n.value;
			e && (r.value = !0, e.scrollTop = e.scrollHeight);
		}
		function b(e) {
			X.activeView = e, (e === "auto" ? Zs(!1) : e === "captured" ? Hs() : e === "samples" ? Vs() : e === "firmware" ? nc() : e === "data" ? rc() : Promise.resolve()).catch((e) => $(e instanceof Error ? e.message : "Refresh failed.", "error"));
		}
		function x(e) {
			X.sampleBucket = e;
		}
		function S(e, t) {
			X.trimBucket = t, X.trimItem = e;
		}
		function C() {
			i.value = X.autoForm.tater_url || "http://127.0.0.1:8501", a.value = "", o.value = !1, X.taterLinkOpen = !0, hn(() => document.querySelector("#pairing-code")?.focus());
		}
		function w() {
			let e = a.value.toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 8);
			a.value = e.length > 4 ? `${e.slice(0, 4)}-${e.slice(4)}` : e;
		}
		async function ee() {
			if (!i.value.trim() || !a.value.trim()) {
				$("Tater address and pairing code are required.", "warning");
				return;
			}
			o.value = await ec(i.value, a.value);
		}
		function te(e) {
			let t = [];
			return e.source_device && t.push(String(e.source_device)), e.wake_word && t.push(String(e.wake_word)), e.max_probability !== null && e.max_probability !== void 0 && t.push(`max ${e.max_probability}`), e.average_probability !== null && e.average_probability !== void 0 && t.push(`avg ${e.average_probability}`), e.detection_profile && t.push(`profile ${String(e.detection_profile).replaceAll("_", " ")}`), e.auto_review_status && t.push(`auto ${String(e.auto_review_status).replaceAll("_", " ")}`), e.vad_max_probability !== null && e.vad_max_probability !== void 0 && t.push(`VAD ${e.vad_max_probability}`), t;
		}
		function ne(e) {
			let t = [];
			e.original_name && e.original_name !== e.saved_as && t.push(`From ${e.original_name}`);
			let n = uc(e.reviewed_at || e.received_at || e.created_at);
			return n && t.push(`Saved ${n}`), e.message && t.push(String(e.message)), e.auto_negative && t.push("Auto-reviewed false positive"), e.auto_positive && t.push("Auto-promoted close miss"), t.join(" · ") || "Training sample";
		}
		function T(e) {
			return String(e.json_url || e.url || e.jsonUrl || "");
		}
		function re(e) {
			return String(e.esphome_json_url || e.esphomeJsonUrl || "");
		}
		function E(e) {
			return String(e.model_url || e.modelUrl || "");
		}
		function ie(e) {
			let t = e.trim().toLowerCase();
			return /^(✓|✅)|success|finished/.test(t) ? "success" : /^(✗|❌)|error|failed|traceback/.test(t) ? "error" : /^(⚠|warning)/.test(t) ? "warning" : /^={4,}|^-----|^=====/.test(t) ? "heading" : "";
		}
		return (e, d) => (U(), W("div", Dc, [
			d[118] ||= G("div", {
				class: "ambient ambient-one",
				"aria-hidden": "true"
			}, null, -1),
			d[119] ||= G("div", {
				class: "ambient ambient-two",
				"aria-hidden": "true"
			}, null, -1),
			G("header", Oc, [G("div", { class: "brand" }, [G("div", {
				class: "brand-mark",
				"aria-hidden": "true"
			}, [G("img", {
				src: bd,
				alt: ""
			})]), d[44] ||= G("div", null, [
				G("span", { class: "eyebrow" }, "Tater tools"),
				G("h1", null, "Wake Word Studio"),
				G("p", null, "Generate voices, curate real recordings, train, and publish.")
			], -1)]), G("div", kc, [d[45] ||= G("span", { class: "live-dot" }, [G("i"), da("Local trainer")], -1), I(X).session.safe_word ? (U(), W("span", Ac, k(I(X).session.safe_word) + " · " + k(I(X).language), 1)) : q("", !0)])]),
			G("nav", jc, [(U(), W(V, null, Lr(s, (e) => G("button", {
				key: e.id,
				type: "button",
				class: O({ active: I(X).activeView === e.id }),
				onClick: (t) => b(e.id)
			}, [
				G("span", Nc, k(e.label), 1),
				G("span", Pc, k(e.short), 1),
				e.id === "captured" && I(X).captured.captured_count ? (U(), W("b", Fc, k(I(X).captured.captured_count), 1)) : q("", !0)
			], 10, Mc)), 64))]),
			G("main", Ic, [I(X).initialized ? (U(), W(V, { key: 1 }, [I(X).activeView === "trainer" ? (U(), W(V, { key: 0 }, [
				d[59] ||= fa("<section class=\"hero training-hero\"><div><span class=\"eyebrow\">Training studio</span><h2>Build a personal wake word</h2><p>Choose a multilingual voice route, check your real samples, then follow the model pipeline live.</p></div><div class=\"step-row\"><span><b>1</b> Phrase</span><span><b>2</b> Samples</span><span><b>3</b> Train</span></div></section>", 1),
				G("section", Rc, [
					G("header", zc, [
						d[47] ||= G("div", { class: "number" }, "1", -1),
						d[48] ||= G("div", null, [G("h3", null, "Phrase + voice"), G("p", null, "The phrase and voice route lock while a session is active.")], -1),
						G("span", { class: O(["pill", I(X).session.safe_word ? "success" : ""]) }, k(I(X).session.safe_word ? `Session · ${I(X).session.safe_word}` : "No session"), 3)
					]),
					G("div", Bc, [
						G("label", Vc, [d[49] ||= G("span", null, "Wake phrase", -1), R(G("input", {
							"onUpdate:modelValue": d[0] ||= (e) => I(X).phrase = e,
							type: "text",
							placeholder: "e.g. \"hey tater\"",
							disabled: !!I(X).session.safe_word || I(Z)("session"),
							onKeyup: d[1] ||= ss((...e) => I(Ls) && I(Ls)(...e), ["enter"])
						}, null, 40, Hc), [[Xo, I(X).phrase]])]),
						G("label", Uc, [
							d[50] ||= G("span", null, "Language", -1),
							R(G("select", {
								"onUpdate:modelValue": d[2] ||= (e) => I(X).language = e,
								disabled: !!I(X).session.safe_word || I(Z)("session")
							}, [(U(!0), W(V, null, Lr(I(X).languages, (e) => (U(), W("option", {
								key: e.code,
								value: e.code
							}, k(e.label), 9, Gc))), 128))], 8, Wc), [[$o, I(X).language]]),
							G("small", null, k(I(Os)), 1)
						]),
						G("label", Kc, [
							d[51] ||= G("span", null, "TTS source", -1),
							R(G("select", {
								"onUpdate:modelValue": d[3] ||= (e) => I(X).ttsMode = e,
								disabled: !!I(X).session.safe_word || I(Z)("session")
							}, [
								G("option", {
									value: "hybrid",
									disabled: !I(X).languages.find((e) => e.code === I(X).language)?.engines?.includes("piper")
								}, "Four-provider ensemble · recommended", 8, Jc),
								G("option", {
									value: "modern",
									disabled: !I(X).languages.find((e) => e.code === I(X).language)?.engines?.some((e) => e !== "piper")
								}, "Modern only · no Piper", 8, Yc),
								G("option", {
									value: "piper",
									disabled: !I(X).languages.find((e) => e.code === I(X).language)?.engines?.includes("piper")
								}, "Piper only · legacy", 8, Xc)
							], 8, qc), [[$o, I(X).ttsMode]]),
							d[52] ||= G("small", null, "Models download once and stay cached.", -1)
						])
					]),
					G("div", Zc, [I(X).session.safe_word ? (U(), W("button", {
						key: 1,
						type: "button",
						class: "button danger",
						disabled: I(Z)("session"),
						onClick: d[5] ||= (...e) => I(Rs) && I(Rs)(...e)
					}, k(I(Z)("session") ? "Stopping…" : I(X).training.running ? "Stop session + training" : "Stop session"), 9, $c)) : (U(), W("button", {
						key: 0,
						type: "button",
						class: "button primary",
						disabled: I(Z)("session") || !I(X).phrase.trim(),
						onClick: d[4] ||= (...e) => I(Ls) && I(Ls)(...e)
					}, k(I(Z)("session") ? "Starting…" : "Start session"), 9, Qc)), G("button", {
						type: "button",
						disabled: !I(X).phrase.trim(),
						onClick: d[6] ||= (...e) => I(zs) && I(zs)(...e)
					}, "System preview", 8, el)])
				]),
				G("section", tl, [
					G("header", nl, [
						d[53] ||= G("div", { class: "number" }, "2", -1),
						d[54] ||= G("div", null, [G("h3", null, "Train wake word"), G("p", null, "Personal positives and reviewed false-wake negatives are automatically included.")], -1),
						G("span", { class: O(["pill", p.value.tone]) }, k(p.value.text), 3)
					]),
					G("div", rl, [
						G("article", null, [d[55] ||= G("span", null, "Positive samples", -1), G("strong", null, k(I(Ts)), 1)]),
						G("article", null, [d[56] ||= G("span", null, "Negative samples", -1), G("strong", null, k(I(Es)), 1)]),
						d[57] ||= G("article", null, [G("span", null, "Training format"), G("strong", { class: "format-value" }, "16 kHz · mono · WAV")], -1)
					]),
					G("div", il, [G("button", {
						type: "button",
						class: "button primary large",
						disabled: !I(X).session.safe_word || I(X).training.running || I(Z)("training-start"),
						onClick: d[7] ||= (...e) => I(oc) && I(oc)(...e)
					}, k(I(X).training.running ? "Training in progress" : "Start training"), 9, al)]),
					G("footer", ol, [d[58] ||= G("span", null, "Training opens the console automatically and continues if the window is closed.", -1), G("button", {
						type: "button",
						disabled: !I(ks),
						onClick: d[8] ||= (e) => I(X).consoleOpen = !0
					}, "Open console", 8, sl)])
				])
			], 64)) : I(X).activeView === "auto" ? (U(), W(V, { key: 1 }, [
				G("section", cl, [d[60] ||= G("div", null, [
					G("span", { class: "eyebrow" }, "False-positive loop"),
					G("h2", null, "Auto Training"),
					G("p", null, "Transcribe captures, sort negatives, recover close misses, retrain on schedule, and publish through Tater.")
				], -1), G("span", { class: O(["pill hero-pill", m.value.tone]) }, k(m.value.text), 3)]),
				G("section", ll, [
					d[68] ||= G("header", { class: "panel-head" }, [G("div", { class: "number" }, "1"), G("div", null, [G("h3", null, "Review rules"), G("p", null, "Conservative local STT keeps uncertain clips in the manual inbox.")])], -1),
					G("div", ul, [
						G("label", null, [R(G("input", {
							"onUpdate:modelValue": d[9] ||= (e) => I(X).autoForm.enabled = e,
							type: "checkbox"
						}, null, 512), [[Zo, I(X).autoForm.enabled]]), d[61] ||= G("span", null, [G("strong", null, "Enable Auto Training"), G("small", null, "Queue eligible wake triggers for local transcription.")], -1)]),
						G("label", null, [R(G("input", {
							"onUpdate:modelValue": d[10] ||= (e) => I(X).autoForm.delete_confirmed_wakes = e,
							type: "checkbox"
						}, null, 512), [[Zo, I(X).autoForm.delete_confirmed_wakes]]), d[62] ||= G("span", null, [G("strong", null, "Delete confirmed good wakes"), G("small", null, "Remove normal triggers when STT confirms the phrase.")], -1)]),
						G("label", null, [R(G("input", {
							"onUpdate:modelValue": d[11] ||= (e) => I(X).autoForm.promote_close_misses = e,
							type: "checkbox"
						}, null, 512), [[Zo, I(X).autoForm.promote_close_misses]]), d[63] ||= G("span", null, [G("strong", null, "Promote confirmed close misses"), G("small", null, "Move verified close misses into positive samples.")], -1)])
					]),
					G("div", dl, [
						G("label", fl, [d[64] ||= G("span", null, "Wake phrase", -1), R(G("input", {
							"onUpdate:modelValue": d[12] ||= (e) => I(X).autoForm.wake_phrase = e,
							type: "text"
						}, null, 512), [[Xo, I(X).autoForm.wake_phrase]])]),
						G("label", pl, [d[65] ||= G("span", null, "STT language", -1), R(G("input", {
							"onUpdate:modelValue": d[13] ||= (e) => I(X).autoForm.language = e,
							type: "text"
						}, null, 512), [[Xo, I(X).autoForm.language]])]),
						G("label", ml, [
							d[66] ||= G("span", null, "STT engine", -1),
							R(G("select", { "onUpdate:modelValue": d[14] ||= (e) => I(X).autoForm.stt_engine = e }, [(U(!0), W(V, null, Lr(I(Ms), (e) => (U(), W("option", {
								key: e.id || e.value,
								value: e.id || e.value
							}, k(e.label || e.name || e.id), 9, hl))), 128))], 512), [[$o, I(X).autoForm.stt_engine]]),
							G("small", null, k(I(Ms).find((e) => (e.id || e.value) === I(X).autoForm.stt_engine)?.description || "Runs locally on this trainer."), 1)
						]),
						G("label", gl, [d[67] ||= G("span", null, "Minimum transcript characters", -1), R(G("input", {
							"onUpdate:modelValue": d[15] ||= (e) => I(X).autoForm.minimum_transcript_chars = e,
							min: "1",
							max: "100",
							type: "number"
						}, null, 512), [[
							Xo,
							I(X).autoForm.minimum_transcript_chars,
							void 0,
							{ number: !0 }
						]])])
					])
				]),
				G("section", _l, [
					d[75] ||= G("header", { class: "panel-head" }, [G("div", { class: "number" }, "2"), G("div", null, [G("h3", null, "Training schedule"), G("p", null, "A run starts only after enough newly reviewed negatives accumulate.")])], -1),
					G("div", vl, [G("label", yl, [d[70] ||= G("span", null, "Run training", -1), R(G("select", { "onUpdate:modelValue": d[16] ||= (e) => I(X).autoForm.schedule_hours = e }, [...d[69] ||= [
						G("option", { value: 0 }, "Manually only", -1),
						G("option", { value: 6 }, "Every 6 hours", -1),
						G("option", { value: 12 }, "Every 12 hours", -1),
						G("option", { value: 24 }, "Every day", -1),
						G("option", { value: 48 }, "Every 2 days", -1),
						G("option", { value: 168 }, "Every week", -1)
					]], 512), [[
						$o,
						I(X).autoForm.schedule_hours,
						void 0,
						{ number: !0 }
					]])]), G("label", bl, [d[71] ||= G("span", null, "Minimum new negatives", -1), R(G("input", {
						"onUpdate:modelValue": d[17] ||= (e) => I(X).autoForm.minimum_new_negatives = e,
						min: "1",
						max: "10000",
						type: "number"
					}, null, 512), [[
						Xo,
						I(X).autoForm.minimum_new_negatives,
						void 0,
						{ number: !0 }
					]])])]),
					G("div", xl, [
						G("article", null, [d[72] ||= G("span", null, "Pending negatives", -1), G("strong", null, k(Number(u.value.pending_negative_count || 0)), 1)]),
						G("article", null, [d[73] ||= G("span", null, "Next check", -1), G("strong", Sl, k(u.value.next_run_at ? I(uc)(u.value.next_run_at) : "Manual"), 1)]),
						G("article", null, [d[74] ||= G("span", null, "Last training", -1), G("strong", Cl, k(u.value.last_train_finished_at ? I(uc)(u.value.last_train_finished_at) : "Never"), 1)])
					])
				]),
				G("section", wl, [
					d[79] ||= G("header", { class: "panel-head" }, [G("div", { class: "number" }, "3"), G("div", null, [G("h3", null, "Publish to Tater"), G("p", null, "Securely activate successful models across every connected satellite.")])], -1),
					G("div", Tl, [G("label", El, [
						d[76] ||= G("span", null, "Trainer public URL", -1),
						R(G("input", {
							"onUpdate:modelValue": d[18] ||= (e) => I(X).autoForm.advertised_base_url = e,
							type: "text",
							placeholder: "Auto-detect LAN address"
						}, null, 512), [[Xo, I(X).autoForm.advertised_base_url]]),
						G("small", null, k(I(X).autoForm.advertised_base_url ? `Configured: ${I(X).autoForm.advertised_base_url}` : `Detected: ${I(X).auto.advertised_base_url || "unavailable"}`), 1)
					]), G("label", Dl, [d[77] ||= G("span", null, "Tater URL", -1), R(G("input", {
						"onUpdate:modelValue": d[19] ||= (e) => I(X).autoForm.tater_url = e,
						type: "text"
					}, null, 512), [[Xo, I(X).autoForm.tater_url]])])]),
					G("div", Ol, [
						G("span", { class: O(["pill", I(js) ? "success" : "warning"]) }, k(I(js) ? `Linked${I(X).auto.trainer_link?.tater_name ? ` · ${I(X).auto.trainer_link.tater_name}` : ""}` : "Not linked"), 3),
						G("button", {
							type: "button",
							class: "button primary",
							disabled: I(Z)("auto"),
							onClick: C
						}, k(I(js) ? "Relink Tater" : "Link Tater"), 9, kl),
						I(js) ? (U(), W("button", {
							key: 0,
							type: "button",
							class: "button danger",
							disabled: I(Z)("auto"),
							onClick: d[20] ||= (...e) => I(tc) && I(tc)(...e)
						}, "Unlink", 8, Al)) : q("", !0)
					]),
					G("div", jl, [G("label", null, [R(G("input", {
						"onUpdate:modelValue": d[21] ||= (e) => I(X).autoForm.notify_satellites = e,
						type: "checkbox"
					}, null, 512), [[Zo, I(X).autoForm.notify_satellites]]), d[78] ||= G("span", null, [G("strong", null, "Activate after successful training"), G("small", null, "Tater applies the new word globally.")], -1)])])
				]),
				G("section", Ml, [G("div", Nl, [
					G("button", {
						type: "button",
						class: "button primary",
						disabled: I(Z)("auto"),
						onClick: d[22] ||= (...e) => I(Qs) && I(Qs)(...e)
					}, "Save Auto Training", 8, Pl),
					G("button", {
						type: "button",
						disabled: I(Z)("auto"),
						onClick: d[23] ||= (e) => I($s)("review_now")
					}, "Review inbox now", 8, Fl),
					G("button", {
						type: "button",
						disabled: I(Z)("auto") || I(X).training.running,
						onClick: d[24] ||= (e) => I($s)("train_now")
					}, "Train now", 8, Il),
					G("button", {
						type: "button",
						disabled: I(Z)("auto") || !I(js),
						onClick: d[25] ||= (e) => I($s)("notify_now")
					}, "Publish current word", 8, Ll)
				]), G("p", Rl, k(f.value), 1)])
			], 64)) : I(X).activeView === "captured" ? (U(), W(V, { key: 2 }, [
				G("section", zl, [d[80] ||= G("div", null, [
					G("span", { class: "eyebrow" }, "Capture review"),
					G("h2", null, "Captured Audio"),
					G("p", null, "Listen to clips from your satellites and turn every real-world event into a better model.")
				], -1), G("span", { class: O(["pill hero-pill", I(X).captured.captured_count ? "warning" : ""]) }, k(I(X).captured.captured_count ? `${I(X).captured.captured_count} waiting` : "Inbox idle"), 3)]),
				G("section", Bl, [G("header", Vl, [
					d[81] ||= G("div", { class: "number" }, "1", -1),
					d[82] ||= G("div", null, [G("h3", null, "Review queue"), G("p", null, "Approve good phrases, keep false positives as negatives, or discard noise.")], -1),
					G("button", {
						type: "button",
						disabled: I(Z)("captured"),
						onClick: d[26] ||= (e) => I(Hs)()
					}, k(I(Z)("captured") ? "Refreshing…" : "Refresh inbox"), 9, Hl)
				]), G("div", Ul, [
					G("article", null, [d[83] ||= G("span", null, "Inbox", -1), G("strong", null, k(I(X).captured.captured_count), 1)]),
					G("article", null, [d[84] ||= G("span", null, "Reviewed negatives", -1), G("strong", null, k(I(Es)), 1)]),
					G("article", null, [d[85] ||= G("span", null, "Personal samples", -1), G("strong", null, k(I(Ts)), 1)])
				])]),
				G("section", Wl, [d[88] ||= G("header", { class: "panel-head" }, [G("div", { class: "number" }, "2"), G("div", null, [G("h3", null, "Listen + sort"), G("p", null, "Metadata remains visible so borderline detections are easy to understand.")])], -1), I(X).captured.items?.length ? (U(), W("div", Kl, [(U(!0), W(V, null, Lr(I(X).captured.items, (e) => (U(), W("article", {
					key: e.saved_as,
					class: "audio-card"
				}, [
					G("header", null, [G("div", null, [G("strong", null, k(e.original_name || e.saved_as), 1), G("small", null, k(I(uc)(e.captured_at || e.received_at)) + " " + k(e.message || ""), 1)]), G("span", { class: O(["pill", I(pc)(e).tone]) }, k(I(pc)(e).label), 3)]),
					te(e).length ? (U(), W("div", ql, [(U(!0), W(V, null, Lr(te(e), (e) => (U(), W("span", { key: e }, k(e), 1))), 128))])) : q("", !0),
					e.transcript ? (U(), W("div", Jl, [d[86] ||= G("b", null, "STT", -1), da(" " + k(e.transcript), 1)])) : q("", !0),
					e.auto_review_guided_transcript ? (U(), W("div", Yl, [d[87] ||= G("b", null, "Guided wake check", -1), da(" " + k(e.auto_review_guided_transcript), 1)])) : q("", !0),
					G("audio", {
						controls: "",
						preload: "none",
						src: I(mc)(e, "captured")
					}, null, 8, Xl),
					G("footer", null, [G("span", null, k(e.saved_as) + " · " + k(I(fc)(e.final_format)), 1), G("div", null, [
						G("button", {
							type: "button",
							disabled: I(Z)("review"),
							onClick: (t) => I(Ks)(e, "approve_personal")
						}, "Add positive", 8, Zl),
						G("button", {
							type: "button",
							disabled: I(Z)("review"),
							onClick: (t) => I(Ks)(e, "mark_negative")
						}, "Mark negative", 8, Ql),
						G("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Z)("review"),
							onClick: (t) => I(Ks)(e, "discard")
						}, "Discard", 8, $l)
					])])
				]))), 128))])) : (U(), W("div", Gl, "No captured audio yet. Clips sent by satellites will appear here."))])
			], 64)) : I(X).activeView === "samples" ? (U(), W(V, { key: 3 }, [
				G("section", eu, [d[89] ||= G("div", null, [
					G("span", { class: "eyebrow" }, "Sample library"),
					G("h2", null, "Current Training Samples"),
					G("p", null, "Audit positives and negatives, trim recordings precisely, and import seed audio.")
				], -1), G("span", tu, k(I(Ts) + I(Es)) + " total", 1)]),
				G("section", nu, [
					G("header", ru, [
						d[92] ||= G("div", { class: "number" }, "1", -1),
						d[93] ||= G("div", null, [G("h3", null, "Saved samples"), G("p", null, "Personal clips are positives. Negative clips are false wakes and hard negatives.")], -1),
						G("div", iu, [G("button", {
							type: "button",
							class: O({ active: I(X).sampleBucket === "personal" }),
							onClick: d[27] ||= (e) => x("personal")
						}, [d[90] ||= da("Personal ", -1), G("b", null, k(I(Ts)), 1)], 2), G("button", {
							type: "button",
							class: O({ active: I(X).sampleBucket === "negative" }),
							onClick: d[28] ||= (e) => x("negative")
						}, [d[91] ||= da("Negative ", -1), G("b", null, k(I(Es)), 1)], 2)])
					]),
					G("div", au, [
						G("button", {
							type: "button",
							disabled: I(Z)("samples"),
							onClick: d[29] ||= (e) => I(Vs)()
						}, "Refresh", 8, ou),
						G("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Z)("review") || I(Ts) === 0,
							onClick: d[30] ||= (e) => I(Ys)("personal")
						}, "Clear positives", 8, su),
						G("button", {
							type: "button",
							class: "button danger ghost",
							disabled: I(Z)("review") || I(Es) === 0,
							onClick: d[31] ||= (e) => I(Ys)("negative")
						}, "Clear negatives", 8, cu)
					]),
					I(As).length ? (U(), W("div", uu, [(U(!0), W(V, null, Lr(c.value, (e) => (U(), W("article", {
						key: e.saved_as,
						class: "audio-card"
					}, [
						G("header", null, [G("div", null, [G("strong", null, k(e.saved_as), 1), G("small", null, k(ne(e)), 1)]), G("div", du, [e.trimmed ? (U(), W("span", fu, "Trimmed")) : q("", !0), G("span", { class: O(["pill", I(X).sampleBucket === "personal" ? "success" : "error"]) }, k(I(X).sampleBucket === "personal" ? "Positive" : "Negative"), 3)])]),
						e.transcript ? (U(), W("div", pu, [d[94] ||= G("b", null, "STT", -1), da(" " + k(e.transcript), 1)])) : q("", !0),
						e.auto_review_guided_transcript ? (U(), W("div", mu, [d[95] ||= G("b", null, "Guided wake check", -1), da(" " + k(e.auto_review_guided_transcript), 1)])) : q("", !0),
						G("audio", {
							controls: "",
							preload: "none",
							src: I(mc)(e, I(X).sampleBucket)
						}, null, 8, hu),
						G("footer", null, [G("span", null, k(I(fc)(e.final_format)), 1), G("div", null, [
							G("button", {
								type: "button",
								onClick: (t) => S(e, I(X).sampleBucket)
							}, "Trim", 8, gu),
							e.trimmed ? (U(), W("button", {
								key: 0,
								type: "button",
								onClick: (t) => I(Js)(e, I(X).sampleBucket)
							}, "Revert", 8, _u)) : q("", !0),
							G("button", {
								type: "button",
								class: "button danger ghost",
								disabled: I(Z)("review"),
								onClick: (t) => I(qs)(e, I(X).sampleBucket)
							}, "Remove", 8, vu)
						])])
					]))), 128))])) : (U(), W("div", lu, "No " + k(I(X).sampleBucket) + " samples saved yet.", 1)),
					l.value > 1 ? (U(), W("div", yu, [
						G("button", {
							type: "button",
							disabled: I(X).samplePage[I(X).sampleBucket] === 0,
							onClick: d[32] ||= (e) => I(X).samplePage[I(X).sampleBucket]--
						}, "Previous", 8, bu),
						G("span", null, "Page " + k(I(X).samplePage[I(X).sampleBucket] + 1) + " of " + k(l.value), 1),
						G("button", {
							type: "button",
							disabled: I(X).samplePage[I(X).sampleBucket] >= l.value - 1,
							onClick: d[33] ||= (e) => I(X).samplePage[I(X).sampleBucket]++
						}, "Next", 8, xu)
					])) : q("", !0)
				]),
				G("section", Su, [
					d[97] ||= G("header", { class: "panel-head" }, [G("div", { class: "number" }, "2"), G("div", null, [G("h3", null, "Manual sample import"), G("p", null, "Optional seed recordings are normalized to the trainer’s required WAV format.")])], -1),
					G("label", Cu, [
						G("input", {
							ref_key: "uploadInput",
							ref: t,
							type: "file",
							multiple: "",
							accept: "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aac,.webm,.opus",
							onChange: d[34] ||= (...e) => I(Us) && I(Us)(...e)
						}, null, 544),
						d[96] ||= G("span", null, [G("strong", null, "Choose one or many audio files"), G("small", null, "WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, and WEBM")], -1),
						G("b", null, k(I(X).selectedFiles.length ? `${I(X).selectedFiles.length} selected` : "Browse"), 1)
					]),
					G("button", {
						type: "button",
						class: "button primary",
						disabled: !I(X).session.safe_word || !I(X).selectedFiles.length || I(Z)("upload"),
						onClick: d[35] ||= (e) => I(Gs)(t.value)
					}, k(I(Z)("upload") ? "Uploading…" : "Upload selected samples"), 9, wu),
					G("div", Tu, [
						G("div", null, [G("strong", null, k(I(X).uploadLabel), 1), G("span", null, k(I(X).uploadProgress) + "%", 1)]),
						G("div", Eu, [G("i", { style: fe({ width: `${I(X).uploadProgress}%` }) }, null, 4)]),
						G("small", null, k(I(X).uploadDetail), 1)
					])
				])
			], 64)) : I(X).activeView === "data" ? (U(), W(V, { key: 4 }, [
				G("section", Du, [d[98] ||= G("div", null, [
					G("span", { class: "eyebrow" }, "Local storage"),
					G("h2", null, "Data Management"),
					G("p", null, "See exactly what the trainer has downloaded, generated, recorded, and produced.")
				], -1), G("span", Ou, k(I(dc)(I(X).managedData.total_size_bytes)) + " total", 1)]),
				G("section", ku, [
					G("header", Au, [
						d[99] ||= G("div", { class: "number" }, "i", -1),
						d[100] ||= G("div", null, [G("h3", null, "Trainer storage"), G("p", null, "Deleting an item is permanent. Required downloads and generated caches will be rebuilt the next time training needs them.")], -1),
						G("button", {
							type: "button",
							disabled: I(Z)("data") || I(Z)("data-delete"),
							onClick: d[36] ||= (e) => I(rc)()
						}, k(I(Z)("data") ? "Scanning…" : "Refresh sizes"), 9, ju)
					]),
					G("div", Mu, [
						G("article", null, [d[101] ||= G("span", null, "Space used", -1), G("strong", Nu, k(I(dc)(I(X).managedData.total_size_bytes)), 1)]),
						G("article", null, [d[102] ||= G("span", null, "Files", -1), G("strong", null, k(Number(I(X).managedData.total_file_count || 0).toLocaleString()), 1)]),
						G("article", null, [d[103] ||= G("span", null, "Individual items", -1), G("strong", null, k(I(X).managedData.items.length), 1)])
					]),
					I(X).training.running ? (U(), W("p", Pu, "Stop the active training session before deleting data.")) : q("", !0)
				]),
				(U(!0), W(V, null, Lr(g.value, (e, t) => (U(), W("section", {
					key: e.name,
					class: "panel data-panel"
				}, [G("header", Fu, [G("div", Iu, k(t + 1), 1), G("div", null, [G("h3", null, k(e.name), 1), G("p", null, k(e.items.length) + " separately managed item" + k(e.items.length === 1 ? "" : "s"), 1)])]), G("div", Lu, [(U(!0), W(V, null, Lr(e.items, (e) => (U(), W("article", {
					key: e.id,
					class: O(["data-row", { empty: !e.file_count }])
				}, [
					G("div", Ru, [
						G("div", zu, [G("strong", null, k(e.label), 1), G("code", null, k(e.location), 1)]),
						G("small", null, k(e.description), 1),
						e.rebuild_note ? (U(), W("span", Bu, k(e.rebuild_note), 1)) : q("", !0)
					]),
					G("div", Vu, [G("strong", null, k(I(dc)(e.size_bytes)), 1), G("span", null, k(Number(e.file_count || 0).toLocaleString()) + " file" + k(e.file_count === 1 ? "" : "s"), 1)]),
					G("button", {
						type: "button",
						class: "button danger ghost",
						disabled: !e.file_count || I(X).training.running || I(Z)("data") || I(Z)("data-delete"),
						onClick: (t) => I(ic)(e)
					}, k(I(Z)("data-delete") ? "Please wait…" : "Delete"), 9, Hu)
				], 2))), 128))])]))), 128)),
				!I(Z)("data") && !I(X).managedData.items.length ? (U(), W("section", Uu, "No managed trainer data was found.")) : q("", !0)
			], 64)) : I(X).activeView === "firmware" ? (U(), W(V, { key: 5 }, [
				G("section", Wu, [d[104] ||= G("div", null, [
					G("span", { class: "eyebrow" }, "Wake-word catalog"),
					G("h2", null, "Trained Wake Words"),
					G("p", null, "Copy a local JSON package URL into Tater to switch every native satellite live.")
				], -1), G("span", { class: O(["pill hero-pill", I(X).wakeWords.length ? "success" : "warning"]) }, k(I(X).wakeWords.length ? `${I(X).wakeWords.length} trained` : "Catalog empty"), 3)]),
				d[109] ||= G("div", { class: "native-notice" }, [G("strong", null, "Tater Native"), G("span", null, "These packages include model metadata and a direct model URL for live satellite updates.")], -1),
				G("section", Gu, [G("header", Ku, [
					d[105] ||= G("div", { class: "number" }, "v1", -1),
					d[106] ||= G("div", null, [G("h3", null, "Published model URLs"), G("p", null, "URLs stay local and are refreshed after each successful run.")], -1),
					G("button", {
						type: "button",
						disabled: I(Z)("firmware"),
						onClick: d[37] ||= (e) => I(nc)()
					}, "Refresh", 8, qu)
				]), I(X).wakeWords.length ? (U(), W("div", Yu, [(U(!0), W(V, null, Lr(I(X).wakeWords, (e) => (U(), W("article", { key: e.key || T(e) }, [G("div", null, [
					G("strong", null, k(e.label || e.name || "Trained wake word"), 1),
					T(e) ? (U(), W("a", {
						key: 0,
						href: T(e),
						target: "_blank",
						rel: "noreferrer"
					}, "JSON · " + k(T(e)), 9, Xu)) : (U(), W("span", Zu, "JSON package URL unavailable")),
					E(e) ? (U(), W("a", {
						key: 2,
						href: E(e),
						target: "_blank",
						rel: "noreferrer"
					}, "Model · " + k(E(e)), 9, Qu)) : q("", !0),
					G("div", $u, [
						e.language ? (U(), W("span", ed, k(e.language), 1)) : q("", !0),
						e.trained_at ? (U(), W("span", td, k(I(uc)(e.trained_at)), 1)) : q("", !0),
						e.recall === void 0 ? q("", !0) : (U(), W("span", nd, "recall " + k(e.recall), 1))
					])
				]), G("button", {
					type: "button",
					disabled: !T(e),
					onClick: (t) => I(ac)(T(e))
				}, "Copy URL", 8, rd)]))), 128))])) : (U(), W("div", Ju, "Train a wake word and its package will appear here."))]),
				d[110] ||= G("div", { class: "native-notice esphome-notice" }, [G("strong", null, "ESPHome"), G("span", null, "Strict micro_wake_word manifest without Tater Native or calibration extensions.")], -1),
				G("section", id, [d[108] ||= G("header", { class: "panel-head" }, [G("div", { class: "number" }, "ESP"), G("div", null, [G("h3", null, "ESPHome JSON"), G("p", null, "Use this URL as the model in an ESPHome micro_wake_word configuration.")])], -1), I(X).wakeWords.length ? (U(), W("div", od, [(U(!0), W(V, null, Lr(I(X).wakeWords, (e) => (U(), W("article", { key: `esphome-${e.key || re(e)}` }, [G("div", null, [
					G("strong", null, k(e.label || e.name || "Trained wake word"), 1),
					re(e) ? (U(), W("a", {
						key: 0,
						href: re(e),
						target: "_blank",
						rel: "noreferrer"
					}, "ESPHome JSON · " + k(re(e)), 9, sd)) : (U(), W("span", cd, "ESPHome package URL unavailable")),
					d[107] ||= G("div", { class: "meta-row" }, [G("span", null, "Schema v2"), G("span", null, "Same TFLite model")], -1)
				]), G("button", {
					type: "button",
					disabled: !re(e),
					onClick: (t) => I(ac)(re(e))
				}, "Copy ESPHome URL", 8, ld)]))), 128))])) : (U(), W("div", ad, "ESPHome links appear after a wake word is trained."))])
			], 64)) : q("", !0)], 64)) : (U(), W("div", Lc, [...d[46] ||= [G("span", { class: "spinner" }, null, -1), G("strong", null, "Connecting to the local trainer…", -1)]]))]),
			(U(), ra(Jn, { to: "body" }, [I(X).consoleOpen ? (U(), W("div", {
				key: 0,
				class: "modal-backdrop console-backdrop",
				onClick: d[39] ||= as((e) => I(X).consoleOpen = !1, ["self"])
			}, [G("section", ud, [G("header", dd, [d[111] ||= G("div", null, [
				G("span", { class: "eyebrow" }, "Live pipeline"),
				G("h2", null, "Training Console"),
				G("p", null, "Closing this window does not interrupt training.")
			], -1), G("div", fd, [
				r.value ? q("", !0) : (U(), W("button", {
					key: 0,
					type: "button",
					class: "console-follow",
					onClick: y
				}, "Jump to latest")),
				G("span", { class: O(["pill", p.value.tone]) }, k(p.value.text), 3),
				G("button", {
					type: "button",
					onClick: d[38] ||= (e) => I(X).consoleOpen = !1
				}, "Close")
			])]), G("pre", {
				ref_key: "consoleLog",
				ref: n,
				class: "console-log",
				onScrollPassive: v
			}, [(U(!0), W(V, null, Lr(h.value, (e, t) => (U(), W("span", {
				key: `${t}-${e}`,
				class: O(ie(e))
			}, k(e), 3))), 128))], 544)])])) : q("", !0)])),
			(U(), ra(Jn, { to: "body" }, [I(X).taterLinkOpen ? (U(), W("div", {
				key: 0,
				class: "modal-backdrop",
				onClick: d[43] ||= as((e) => I(X).taterLinkOpen = !1, ["self"])
			}, [G("section", pd, [G("header", md, [G("div", null, [
				d[112] ||= G("span", { class: "eyebrow" }, "Secure pairing", -1),
				G("h2", null, k(o.value ? "Tater linked" : "Link Tater"), 1),
				G("p", null, k(o.value ? "This trainer can securely publish wake-word updates." : "Enter the short-lived code shown in Tater Voice Settings."), 1)
			]), G("button", {
				type: "button",
				onClick: d[40] ||= (e) => I(X).taterLinkOpen = !1
			}, "Close")]), o.value ? (U(), W("div", hd, [
				d[113] ||= G("i", null, "✓", -1),
				G("strong", null, "Successfully linked" + k(I(X).auto.trainer_link?.tater_name ? ` to ${I(X).auto.trainer_link.tater_name}` : ""), 1),
				d[114] ||= G("span", null, "The private link key is stored locally and is never displayed.", -1)
			])) : (U(), W("div", gd, [
				G("label", _d, [d[115] ||= G("span", null, "Tater address", -1), R(G("input", {
					"onUpdate:modelValue": d[41] ||= (e) => i.value = e,
					type: "text"
				}, null, 512), [[Xo, i.value]])]),
				G("label", vd, [d[116] ||= G("span", null, "Tater pairing code", -1), R(G("input", {
					id: "pairing-code",
					"onUpdate:modelValue": d[42] ||= (e) => a.value = e,
					class: "pairing-code",
					maxlength: "9",
					placeholder: "ABCD-EFGH",
					autocomplete: "off",
					onInput: w
				}, null, 544), [[Xo, a.value]])]),
				d[117] ||= G("small", null, "In Tater, open Voice Settings → Wake Word Trainer → Link Trainer.", -1),
				G("button", {
					type: "button",
					class: "button primary",
					disabled: I(Z)("link"),
					onClick: ee
				}, k(I(Z)("link") ? "Linking securely…" : "Link Tater"), 9, yd)
			]))])])) : q("", !0)])),
			K(Ec),
			K($a, { name: "toast" }, {
				default: Dn(() => [I(X).toast.message ? (U(), W("div", {
					key: 0,
					class: O(["toast", I(X).toast.tone]),
					role: "status"
				}, k(I(X).toast.message), 3)) : q("", !0)]),
				_: 1
			})
		]));
	}
}), Cd = document.getElementById("trainer-app");
if (!Cd) throw Error("Missing #trainer-app mount point");
ds(Sd).mount(Cd);
//#endregion
